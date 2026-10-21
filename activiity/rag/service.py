"""AgenticRagService — ReAct agent with 2 tools (query_by_ua + global).

v6 changes vs v5:
  - verify_context tool REMOVED. Escalation is now AUTOMATIC inside the
    service: query_by_ua re-queries up to CFG.retry_empty times when it
    returns zero nodes (transient embed/network failures), then falls back
    to query_global. Zero extra LLM round-trips; the router does at most
    ONE tool call before "Answer: FIN".
  - Router token cap (CFG.router_max_tokens) and temperature 0.1 for
    faster, more deterministic routing.
  - No classifier fallback, no naive rescue: if even query_global returns
    nothing, the answer is a refusal.

v5 changes vs v4:
  - New verify_context tool: deterministic EMPTY / INSUFFICIENT / OK verdict
    on the nodes captured so far, so the router loops with a hard rule:
    query_by_ua (batched) -> verify -> ONE retry of query_by_ua ->
    verify -> query_global (escalade) -> verify -> refusal if EMPTY.
    (Superseded by the automatic escalation in v6.)
  - Two refusal levels: (1) agentic refusal kept when the question is
    out-of-scope (no tool call), (2) final refusal when the synthesized
    answer comes back empty.

v4 changes vs v3:
  - SLM-optimized system prompt: shorter, example-based UA descriptions,
    explicit disambiguation rules (CONFLIT→UA-4, MOTIVATION→UA-3, etc.)
  - Direct routing fallback: if ReAct agent times out or returns empty,
    a lightweight classification prompt (single LLM call) determines the
    correct UA and dispatches directly — no multi-step agent loop.
  - QUERY_BY_UA_DESCRIPTION includes the full UA list for better tool schema.

v3 changes vs v2:
  - Reduced from 11 tools (10 per-UA + global) to 2 tools.
    The agent now specifies ua_id as a parameter, not by tool selection.
  - Added ContextCompressor postprocessor (sentence-level relevance filtering).
  - Removed refusal guard (prompt fix eliminated false refusals).
  - Multi-tool firing: agent can call query_by_ua multiple times per step.
  - Captures tool_kwargs for eval routing inference.

Public contract (matches NaiveRagService for eval-runner compatibility):

    async def achat_with_trace(question: str) -> AgenticResult
    where AgenticResult exposes:
      .answer, .tools_called, .retrieved_nodes,
      .tokens_in, .tokens_out, .iterations,
      .model_name, .embed_model, .ttft_s
"""
from __future__ import annotations
import re, time
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex, Settings as LISettings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import (
    MetadataFilters, MetadataFilter, FilterOperator,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

from activiity.config import CFG, UA_IDS, collection_name, embed_model_name
from activiity.providers.factory import build_llm, build_synth_llm, build_embed
from activiity.prompts.defaults_fr import (
    SYSTEM_FR, QA_FR, REFINE_FR, REFUSAL_FR,
    QUERY_BY_UA_DESCRIPTION,
)


try:
    from pydantic import ConfigDict, Field, PrivateAttr
except Exception:  # noqa: BLE001
    ConfigDict = None  # type: ignore
    Field = None  # type: ignore
    PrivateAttr = None  # type: ignore


def _json_safe(value):
    """Recursively convert tool-kwargs to plain Python types.

    llama-index wraps list arguments (e.g. ua_ids) in an AttributedList, a
    list subclass whose __init__ signature breaks dataclasses.asdict()
    (``type(obj)(generator)`` passes 2 positional args). Eval serializes
    tool_kwargs to JSON/CSV, so coerce every value here, at the source.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _node_text_meta(node) -> tuple[str, dict]:
    """Normalize a NodeWithScore / Node to (text, metadata)."""
    node_obj = getattr(node, "node", None) or node
    md = getattr(node_obj, "metadata", None)
    if not isinstance(md, dict):
        md = {}
    text = getattr(node_obj, "text", "") or ""
    return str(text), md


def nodes_verdict(
    nodes: list,
    expected_uas: list[str] | None = None,
    post_global: bool = False,
) -> tuple[str, dict]:
    """Deterministic verdict on a set of retrieved nodes (v5 verify_context).

    Returns (verdict, info) where verdict is one of:
      EMPTY        — no usable node text at all
      INSUFFICIENT — too few usable nodes, too little retained text, OR one
                     of the planned UAs is not covered (post-rerank/compression)
      OK           — enough context to synthesize

    `expected_uas` are the management units the router planned to cover
    (from query_by_ua calls); a planned UA with no retrieved chunk is
    reported as "missing" and turns the verdict INSUFFICIENT.

    `post_global` is the query_global escalation mode: after the unfiltered
    search (the last resort) the verdict only depends on empty vs non-empty —
    thresholds and UA coverage are ignored, so any content retrieved globally
    is handed to synthesis.

    The info dict carries the counts so the router (and logs) can see why.
    """
    usable = []
    for n in nodes or []:
        text, md = _node_text_meta(n)
        if text.strip():
            usable.append((text.strip(), md))
    covered = sorted({md.get("ua_id") for _, md in usable if md.get("ua_id")})
    expected = sorted({u for u in (expected_uas or []) if u})
    missing = [u for u in expected if u not in covered]
    info = {
        "nodes": len(nodes or []),
        "usable": len(usable),
        "chars": sum(len(t) for t, _ in usable),
        "uas": covered,
        "expected_uas": expected,
        "missing_uas": missing,
    }
    if not usable:
        verdict = "EMPTY"
    elif post_global:
        verdict = "OK"
    elif (
        len(usable) < CFG.verify_min_nodes
        or info["chars"] < CFG.verify_min_chars
        or missing
    ):
        verdict = "INSUFFICIENT"
    else:
        verdict = "OK"
    info["verdict"] = verdict
    return verdict, info


def _empty_answer(answer: str) -> bool:
    """True when a synthesized answer carries no usable content (2nd refusal)."""
    cleaned = (answer or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in ("fin", "answer: fin", "answer:"):
        return True
    return False


def _is_plausible_refusal(text: str) -> bool:
    """True when router text (no tool call) looks like a deliberate OOS refusal.

    Design C: the router never writes the final answer. The one exception is
    an out-of-scope question where it refuses WITHOUT calling a tool — we
    keep that refusal verbatim for the eval. Everything else (garbage, empty,
    truncated reasoning) is replaced by the canonical REFUSAL_FR. A long
    repetitive text (e.g. a model that looped on "thought") is NOT a refusal.
    """
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return False
    if len(cleaned) > 400:
        return False
    lowered = cleaned.lower()
    markers = (
        "je ne dispose pas", "ne dispose pas", "pas d'information",
        "ne trouve pas", "aucune information", "information manquante",
        "hors sujet", "hors du sujet", "ne peut pas répondre",
        "je ne peux pas répondre", "pas en mesure", "ne relève pas",
        "reformuler", "absente du corpus", "ne figure pas", "base activiity",
    )
    return any(m in lowered for m in markers)


# ---------------------------------------------------------------------------
# Node capture postprocessor (unchanged from v2 — stable across LlamaIndex versions)
# ---------------------------------------------------------------------------

def _agentic_log(msg: str) -> None:
    """Emit an [agentic] debug line only when ACTIVIITY_VERBOSE=1.

    Default OFF: eval runs then show only the runner's progress output.
    """
    if CFG.verbose:
        print(f"[agentic] {msg}", flush=True)


def _norm_question(q: str | None) -> str:
    """Canonical key for per-run state (capture buckets, merge cache).

    The router may lightly normalize the question it passes to tools
    (collapse whitespace, strip trailing spaces before '?'), so capture and
    read-back must key on the same normalized form to avoid empty buckets.
    """
    return " ".join((q or "").split()) or "__no_key__"


class _CapturePostprocessor(BaseNodePostprocessor):
    """Postprocessor that appends retrieved nodes into a per-question buffer.

    Thread-safe across CONCURRENT runs: the eval runs many pairs against one
    service instance (asyncio.gather + workers). A single shared
    ``service._captured_nodes`` list gets reset/mixed between coroutines, so
    nodes are keyed by the query string instead. Each run reads back the
    bucket for its own question.
    """

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    if Field is not None:
        service: object = Field(exclude=True, repr=False)
    else:  # pragma: no cover
        service: object

    def _bucket(self, query_bundle) -> str:
        q = getattr(query_bundle, "query_str", None)
        return _norm_question(q)

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        try:
            bucket = self._bucket(query_bundle)
            self.service._nodes_by_question.setdefault(bucket, []).extend(list(nodes))
        except Exception:  # noqa: BLE001
            pass
        return nodes

    async def _apostprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        try:
            bucket = self._bucket(query_bundle)
            self.service._nodes_by_question.setdefault(bucket, []).extend(list(nodes))
        except Exception:  # noqa: BLE001
            pass
        return nodes


# ---------------------------------------------------------------------------
# Context compression postprocessor — sentence-level relevance filtering
# ---------------------------------------------------------------------------

class ContextCompressor(BaseNodePostprocessor):
    """Filters each node to keep only sentences most relevant to the query.

    Strategy: simple keyword-overlap scoring (no extra LLM call).
    For each sentence in the node, count how many query tokens appear in it.
    Keep sentences scoring above the median. This reduces noise while
    preserving factual content for the synthesizer.
    """

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    if Field is not None:
        keep_ratio: float = Field(default=0.6)
    else:
        keep_ratio: float = 0.6

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple French-aware tokenizer: lowercase, split on non-alpha."""
        return {w for w in re.split(r"[^a-zàâäéèêëïîôùûüÿçœæ]+", text.lower()) if len(w) > 2}

    def _compress_node(self, node_text: str, query_tokens: set[str]) -> str:
        if not query_tokens or not node_text:
            return node_text
        sentences = re.split(r"(?<=[.!?])\s+", node_text)
        if len(sentences) <= 2:
            return node_text
        scored = []
        for s in sentences:
            s_tokens = self._tokenize(s)
            overlap = len(query_tokens & s_tokens)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_n = max(1, int(len(sentences) * self.keep_ratio))
        kept = scored[:keep_n]
        kept.sort(key=lambda x: sentences.index(x[1]))
        return " ".join(s for _, s in kept)

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        if not nodes or not CFG.use_context_compression:
            return nodes
        query = ""
        if query_bundle is not None:
            query = getattr(query_bundle, "query_str", "") or ""
        if not query:
            return nodes
        query_tokens = self._tokenize(query)
        for n in nodes:
            node_obj = getattr(n, "node", n)
            text = getattr(node_obj, "text", "")
            if text and len(text) > 200:
                compressed = self._compress_node(text, query_tokens)
                try:
                    node_obj.text = compressed
                except Exception:  # noqa: BLE001
                    pass
        return nodes

    async def _apostprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        return self._postprocess_nodes(nodes, query_bundle)


# ---------------------------------------------------------------------------
# Rerankers (unchanged from v2)
# ---------------------------------------------------------------------------

class HFInferenceReranker(BaseNodePostprocessor):
    """HF Inference API reranker as a LlamaIndex node postprocessor."""

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    top_n: int
    api_key: str | None = None

    if PrivateAttr is not None:
        _client = PrivateAttr(default=None)

    def model_post_init(self, __context):  # noqa: D401
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key or None,
        )

    def _score_pair(self, query: str, passage: str) -> float:
        text = f"query: {query}\npassage: {passage}"
        res = self._client.text_classification(text, model=self.model)
        if isinstance(res, list) and res:
            first = res[0]
            if isinstance(first, dict) and "score" in first:
                return float(first["score"])
            score = getattr(first, "score", None)
            if score is not None:
                return float(score)
        return 0.0

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        if not nodes:
            return nodes
        query = ""
        if query_bundle is not None:
            query = getattr(query_bundle, "query_str", "") or ""
        if not query:
            return nodes
        scored = []
        for n in nodes:
            node_obj = getattr(n, "node", None)
            if node_obj is not None:
                passage = getattr(node_obj, "text", "")
            else:
                passage = getattr(n, "text", "")
            score = self._score_pair(query, passage)
            try:
                n.score = score
            except Exception:  # noqa: BLE001
                pass
            scored.append(n)
        scored.sort(key=lambda x: x.score if getattr(x, "score", None) is not None else 0.0, reverse=True)
        return scored[: self.top_n]


class OpenRouterReranker(BaseNodePostprocessor):
    """OpenRouter reranker using the /rerank endpoint."""

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    top_n: int
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        if not nodes:
            return nodes
        query = ""
        if query_bundle is not None:
            query = getattr(query_bundle, "query_str", "") or ""
        if not query:
            return nodes
        passages: list[str] = []
        for n in nodes:
            node_obj = getattr(n, "node", None)
            if node_obj is not None:
                passage = getattr(node_obj, "text", "")
            else:
                passage = getattr(n, "text", "")
            passages.append(passage)
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "query": query,
                "documents": passages,
                "top_n": min(self.top_n, len(passages)),
                "return_documents": False,
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(f"{self.base_url}/rerank", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            scored = list(nodes)
            for r in results:
                idx = r.get("index")
                score = r.get("relevance_score")
                if idx is None or score is None:
                    continue
                try:
                    scored[idx].score = float(score)
                except Exception:  # noqa: BLE001
                    pass
            scored.sort(key=lambda x: x.score if getattr(x, "score", None) is not None else 0.0, reverse=True)
            return scored[: self.top_n]
        except Exception as e:  # noqa: BLE001
            _agentic_log(f"openrouter reranker failed ({e}); continuing without")
            return nodes


class QuotaReranker(BaseNodePostprocessor):
    """Per-unit rerank quota — never a global top_n.

    The wrapped reranker scores ALL retrieved candidates in one batch; this
    postprocessor then slices the best ``per_unit_n`` nodes per ``ua_id``.
    Multi-unit retrievals (one batched query_by_ua call) therefore keep full
    per-unit coverage instead of losing weak units to cross-unit pruning.
    """

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    per_unit_n: int
    reranker: BaseNodePostprocessor

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        if not nodes:
            return nodes
        try:
            self.reranker.top_n = len(nodes)  # score everything, truncate per unit after
        except Exception:  # noqa: BLE001
            pass
        scored = self.reranker._postprocess_nodes(nodes, query_bundle)
        if not scored:
            return nodes
        groups: dict[str, list] = {}
        for n in scored:
            node_obj = getattr(n, "node", None)
            md = getattr(node_obj, "metadata", None) if node_obj is not None else None
            md = md or {}
            ua = md.get("ua_id") or "GLOBAL"
            groups.setdefault(ua, []).append(n)
        out: list = []
        for ua, items in groups.items():
            keep = self.per_unit_n if ua != "GLOBAL" else max(self.per_unit_n, 8)
            out.extend(items[:keep])
        return out or scored


class PerUaRetriever(BaseRetriever):
    """Retrieves a per-unit candidate budget and merges the results.

    A single ANN over the union of units can be dominated by the most-similar
    unit (e.g. 18 of 20 hits from one UA), starving the others *before* any
    reranking. Retrieving ``per_ua`` candidates per unit guarantees every
    requested unit is represented, so the per-unit rerank quota can never be
    starved. Still ONE tool call — N cheap filtered ANN searches.
    """

    def __init__(self, index, ua_ids, per_ua, **kwargs):
        super().__init__(**kwargs)
        self._index = index
        self._ua_ids = list(ua_ids)
        self._per_ua = per_ua

    def _retrieve(self, query_bundle):  # type: ignore[override]
        query = query_bundle.query_str
        merged: list = []
        for ua in self._ua_ids:
            filt = MetadataFilters(filters=[
                MetadataFilter(key="ua_id", value=ua, operator=FilterOperator.EQ),
            ])
            retriever = self._index.as_retriever(
                similarity_top_k=self._per_ua, filters=filt,
            )
            merged.extend(retriever.retrieve(query))
        return merged


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgenticResult:
    answer: str
    tools_called: list[str]
    tool_kwargs: list[dict]            # NEW: kwargs per tool call for eval routing
    units_called: list[str]            # distinct UA IDs queried (e.g. ['UA-3','UA-7'] or ['GLOBAL'])
    retrieved_nodes: list              # NodeWithScore (union across tool calls)
    tokens_in: int
    tokens_out: int
    iterations: int
    model_name: str
    embed_model: str
    ttft_s: float | None


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class AgenticRagService:
    def __init__(self):
        self.llm = build_llm()
        self.synth_llm = build_synth_llm()
        self.embed = build_embed()
        LISettings.llm = self.llm
        LISettings.embed_model = self.embed

        # Optional reranker — per-unit quota so multi-UA retrievals keep coverage
        self.reranker = None
        if CFG.use_reranker:
            try:
                if CFG.reranker_provider in ("hf", "hf-inference"):
                    inner = HFInferenceReranker(
                        model=CFG.reranker_model,
                        api_key=CFG.hf_token,
                        top_n=CFG.rerank_keep,
                    )
                    _agentic_log(f"reranker enabled (hf): {CFG.reranker_model}")
                elif CFG.reranker_provider == "openrouter":
                    inner = OpenRouterReranker(
                        model=CFG.reranker_model,
                        api_key=CFG.openrouter_api_key,
                        top_n=CFG.rerank_keep,
                    )
                    _agentic_log(f"reranker enabled (openrouter): {CFG.reranker_model}")
                else:
                    from llama_index.postprocessor.flag_embedding_reranker import (
                        FlagEmbeddingReranker,
                    )
                    inner = FlagEmbeddingReranker(
                        model=CFG.reranker_model, top_n=CFG.rerank_keep,
                    )
                    _agentic_log(f"reranker enabled: {CFG.reranker_model}")
                self.reranker = QuotaReranker(
                    per_unit_n=CFG.rerank_keep, reranker=inner,
                )
            except Exception as e:  # noqa: BLE001
                _agentic_log(f"reranker unavailable ({e}); continuing without")

        # Context compressor (improvement #6)
        self.compressor = ContextCompressor(keep_ratio=0.65) if CFG.use_context_compression else None

        self.token_counter = TokenCountingHandler()
        LISettings.callback_manager = CallbackManager([self.token_counter])

        # Open the Qdrant collection
        coll = collection_name(embed_model_name())
        client = QdrantClient(url=CFG.qdrant_url, timeout=60)
        aclient = AsyncQdrantClient(url=CFG.qdrant_url, timeout=60)
        if not client.collection_exists(coll):
            raise RuntimeError(
                f"Qdrant collection {coll!r} not found — "
                f"run `python -m activiity.ingest.cli` first."
            )
        self.client = client
        self.aclient = aclient
        self.collection = coll
        self.vstore = QdrantVectorStore(
            client=client, aclient=aclient, collection_name=coll,
        )
        self.index = VectorStoreIndex.from_vector_store(self.vstore)

        # Synthesis templates
        self._qa_tmpl = PromptTemplate(QA_FR)
        self._refine_tmpl = PromptTemplate(REFINE_FR)

        # Per-run node capture, keyed by question so CONCURRENT runs (eval:
        # many pairs on one service instance) never clobber each other. Each
        # run reads the bucket for its own question.
        self._nodes_by_question: dict[str, list] = {}
        self._capture_pp = _CapturePostprocessor(service=self)

        # Guards repeated query_by_ua calls for the same question within a
        # single run (see _dispatch_by_ua). Keyed by canonical question.
        self._ua_merge_cache: dict = {}

        # Set per question when query_global is called (see _dispatch_global):
        # relaxes the coverage requirement of the informational post-agent
        # verdict. Keyed by question to survive concurrent runs.
        self._escalated_by_question: dict[str, bool] = {}

        # Pre-create per-UA query engines for the dispatch function
        self._ua_engines = {}
        for ua_id in UA_IDS:
            self._ua_engines[ua_id] = self._build_ua_engine([ua_id])

        # Build global engine separately
        self._global_engine = self._build_global_engine()

        # Build agent with 2 tools
        self.agent = self._build_agent()

        # Cached display names
        self._llm_name = getattr(self.llm, "model", "unknown")
        self._embed_name = getattr(
            self.embed, "model_name",
            getattr(self.embed, "model", "unknown"),
        )

    # ------------------------------------------------------------------
    # Query engine builders
    # ------------------------------------------------------------------
    def _tool_context_text(self, nodes: list, max_chars: int = 200) -> str:
        """Router-facing text for a tool result (design C).

        The engines use response_mode="no_text": they only retrieve and
        post-process nodes (compressor + capture), they never synthesize.
        This builds a compact, tagged preview of the retrieved nodes so the
        router can judge usefulness/coverage; the real synthesis happens once
        over the full post-processed nodes afterwards.

        The body is truncated hard (~1.4 KB total): the router's only job is
        "content exists -> Answer: FIN", and every token here is sent to the
        router LLM on EVERY query (this was the dominant cost of the p95
        latency). The synthesis model gets the full nodes separately.
        """
        parts = []
        for n in nodes:
            node_obj = getattr(n, "node", None) or n
            md = getattr(node_obj, "metadata", None) or {}
            text = getattr(node_obj, "text", "") or ""
            ua = md.get("ua_id", "?")
            snippet = text.strip()[:max_chars]
            parts.append(f"[{ua}] {snippet}" if snippet else f"[{ua}]")
        if not parts:
            return "(aucun contenu récupéré)"
        counts: dict[str, int] = {}
        for p in parts:
            ua = p.split("]")[0][1:]
            counts[ua] = counts.get(ua, 0) + 1
        coverage = ", ".join(f"{ua}×{n}" for ua, n in counts.items())
        header = f"{len(parts)} extrait(s) récupéré(s) ({coverage})."
        body = "\n\n".join(parts)
        budget = 1400 - len(header) - 1
        if len(body) > budget:
            body = body[:budget] + " …"
        return f"{header}\n{body}"

    def _make_postprocessors(self) -> list:
        """Build the postprocessor chain: [reranker, compressor, capture]."""
        pps = []
        if self.reranker:
            pps.append(self.reranker)
        if self.compressor:
            pps.append(self.compressor)
        pps.append(self._capture_pp)
        return pps

    def _build_ua_engine(self, ua_ids: list[str]):
        """Query engine with a guaranteed per-unit retrieval budget.

        Each requested UA contributes ``per_ua`` candidates via its own
        filtered ANN search (merged in order), then the postprocessor chain
        applies the per-unit rerank quota, compression and capture. top_k
        never scales by the union: a single union ANN is dominated by the
        most-similar unit and starves the others.
        """
        per_ua = CFG.rerank_top_k if self.reranker else CFG.top_k_per_ua
        if self.reranker:
            per_ua = max(1, min(per_ua, CFG.rerank_max_total // len(ua_ids)))
        retriever = PerUaRetriever(
            index=self.index, ua_ids=list(ua_ids), per_ua=per_ua,
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(response_mode="no_text"),
            node_postprocessors=self._make_postprocessors(),
        )

    def _build_global_engine(self):
        """Build the global (unfiltered) query engine."""
        top_k = CFG.rerank_top_k if self.reranker else CFG.top_k_global
        return self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="no_text",
            node_postprocessors=self._make_postprocessors(),
        )

    # ------------------------------------------------------------------
    # Tool dispatch functions
    # ------------------------------------------------------------------
    def _dispatch_by_ua(self, ua_ids: list[str], question: str) -> str:
        """Dispatch a query to the per-UA engines for one or more UAs.

        v6 escalation (automatic, no extra LLM round-trips): if the filtered
        search returns ZERO nodes, retry up to CFG.retry_empty times (handles
        transient embed/network failures), then fall back to query_global.
        """
        if isinstance(ua_ids, str):
            ua_ids = [ua_ids]
        valid = [u for u in ua_ids if u in self._ua_engines]
        if not valid:
            return (
                f"UAs inconnues : {ua_ids}. UAs disponibles : "
                f"UA-1 à UA-9. Utilise query_global pour une recherche globale."
            )
        # Guard: the router sometimes re-queries UA(s) it already asked for,
        # even under a slightly different reformulation. Key the merge on the
        # normalized question so repeated calls fold into a single filtered
        # search over the union (design C: one batched call). Keying by
        # question also keeps concurrent eval runs isolated.
        canonical = _norm_question(question)
        _agentic_log(f"query_by_ua tool question={question!r} canonical={canonical!r}")
        cached = self._ua_merge_cache.get(canonical)
        if cached is not None and set(valid) <= set(cached[0]):
            return cached[1]
        merged = list(dict.fromkeys((cached[0] if cached else []) + valid))
        engine = self._build_ua_engine(merged)

        nodes: list = []
        attempts = 1 + CFG.retry_empty
        for attempt in range(attempts):
            try:
                response = engine.query(canonical)
                if response.source_nodes:
                    nodes = list(response.source_nodes)
                    break
            except Exception as e:  # noqa: BLE001
                _agentic_log(f"query_by_ua attempt {attempt + 1}/{attempts} failed ({e})")
        if not nodes:
            # Automatic fallback: unfiltered global search (last resort).
            _agentic_log("query_by_ua empty after retries — falling back to query_global")
            self._escalated_by_question[canonical] = True
            try:
                nodes = list(self._global_engine.query(canonical).source_nodes)
            except Exception as e:  # noqa: BLE001
                _agentic_log(f"query_global fallback failed ({e})")

        text = self._tool_context_text(nodes)
        self._ua_merge_cache[canonical] = (merged, text)
        return text

    def _requested_uas(self, question: str) -> list[str]:
        """The UAs the router planned to cover in this run.

        Read from the query_by_ua merge cache entry keyed by THIS run's
        question (concurrent runs each keep their own entry).
        """
        cached = self._ua_merge_cache.get(_norm_question(question))
        if not isinstance(cached, tuple) or not cached:
            return []
        merged = cached[0]
        return list(merged) if isinstance(merged, list) else []

    # ------------------------------------------------------------------
    # Agent builder
    # ------------------------------------------------------------------
    def _build_agent(self) -> ReActAgent:
        # ONLY query_by_ua is exposed to the router. query_global is NOT a
        # router tool: the escalation (retry ×CFG.retry_empty then unfiltered
        # global search) is automatic inside _dispatch_by_ua, so the router
        # physically cannot call query_global directly. The router ends with
        # "Answer: FIN" and the synthesis model writes the response (design C).
        query_by_ua_tool = FunctionTool.from_defaults(
            fn=self._dispatch_by_ua,
            name="query_by_ua",
            description=QUERY_BY_UA_DESCRIPTION,
        )

        tools = [query_by_ua_tool]

        base_kwargs = dict(
            name="Activiity",
            description="Agent RAG management/coaching FR (Fiches Protocole UA-1..UA-9).",
            tools=tools,
            llm=self.llm,
            system_prompt=SYSTEM_FR,
            timeout=180,
            verbose=False,
            # CFG.max_iterations =  = the verified-routing sequence
            # (query_by_ua, verify, retry, verify, global, verify, Answer: FIN)
            # with headroom; "generate" turns a cap trip into a response, not
            # an error. Misses are recovered by the naive global pipeline.
            early_stopping_method="generate",
        )
        return ReActAgent(**base_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def achat_with_trace(self, question: str) -> AgenticResult:
        self.token_counter.reset_counts()
        # Per-question run state: isolated from other concurrent runs sharing
        # this service instance (eval runs many pairs with asyncio.gather).
        canonical = _norm_question(question)
        self._nodes_by_question.pop(canonical, None)
        self._ua_merge_cache.pop(canonical, None)
        self._escalated_by_question.pop(canonical, None)

        from llama_index.core.agent.workflow import ToolCall
        tools_called: list[str] = []
        tool_kwargs_list: list[dict] = []

        t0 = time.perf_counter()
        agent_answer = ""
        agent_failed = False

        async def _run_agent_once() -> tuple[str, list[str], list[dict], bool]:
            calls: list[str] = []
            kwargs: list[dict] = []
            try:
                handler = self.agent.run(
                    user_msg=question, max_iterations=CFG.max_iterations
                )
                async for ev in handler.stream_events():
                    if isinstance(ev, ToolCall):
                        calls.append(ev.tool_name)
                        kwargs.append(
                            _json_safe(getattr(ev, "tool_kwargs", {}))
                            if hasattr(ev, "tool_kwargs")
                            else {}
                        )
                result = await handler
                resp = getattr(result, "response", None)
                if resp is not None and getattr(resp, "content", None):
                    text = resp.content
                else:
                    text = str(result)
                return text, calls, kwargs, False
            except Exception as e:  # noqa: BLE001
                return f"[agent_error] {type(e).__name__}: {e}", calls, kwargs, True

        agent_answer, tools_called, tool_kwargs_list, agent_failed = (
            await _run_agent_once()
        )

        # Transient-model guard: sometimes (notably under concurrent load) the
        # router emits garbage with NO tool call at all ("thought thought ..."
        # loops, truncated reasoning). Sequential retries recover cleanly, so
        # retry ONCE before falling back to a refusal. A plausible refusal
        # (OOS) is NOT retried — it's kept verbatim.
        if (
            not agent_failed
            and not tools_called
            and not _is_plausible_refusal(agent_answer)
            and (agent_answer or "").strip()
        ):
            _agentic_log("router returned no tool call + no refusal; retrying once")
            self._nodes_by_question.pop(canonical, None)
            self._ua_merge_cache.pop(canonical, None)
            self._escalated_by_question.pop(canonical, None)
            agent_answer, tools_called, tool_kwargs_list, agent_failed = (
                await _run_agent_once()
            )

        # Debug: check routing was actually useful.
        # Single-step ReAct: query_by_ua does its own retry/global escalation
        # internally (up to CFG.retry_empty tries, then query_global), so the
        # router makes at most ONE tool call. Intermediate text is never the
        # final answer and is not logged.
        _agentic_log(f"agent result: tools={tools_called}")

        for kw in tool_kwargs_list:
            tq = (kw or {}).get("question")
            if not tq:
                continue
            tk = _norm_question(tq)
            if tk == canonical:
                continue
            if tk in self._nodes_by_question:
                self._nodes_by_question.setdefault(canonical, []).extend(
                    self._nodes_by_question.pop(tk)
                )
            if tk in self._ua_merge_cache:
                ent = self._ua_merge_cache.pop(tk)
                if canonical in self._ua_merge_cache:
                    uas = list(dict.fromkeys(list(self._ua_merge_cache[canonical][0]) + list(ent[0])))
                    self._ua_merge_cache[canonical] = (uas, self._ua_merge_cache[canonical][1] or ent[1])
                else:
                    self._ua_merge_cache[canonical] = ent
            if tk in self._escalated_by_question:
                self._escalated_by_question[canonical] = self._escalated_by_question.pop(tk)

        # Design C: the router never writes the final answer — it only routes
        # (tool calls) and signals refusal. If a tool returned chunks, the
        # synthesis model writes the answer once over the post-processed
        # nodes at the end; router-generated text is discarded.

        captured = list(self._nodes_by_question.get(canonical, []))
        _agentic_log(
            f"post-agent read canonical={canonical!r} "
            f"nodes_keys={list(self._nodes_by_question)} "
            f"cache_keys={list(self._ua_merge_cache)}"
        )

        # Deterministic verification of what the agent actually retrieved
        # (informational — the query_global tool is the ONLY escalation; if
        # nothing was retrieved, the answer stays a refusal, no rescue).
        verdict, vinfo = nodes_verdict(
            captured,
            expected_uas=self._requested_uas(question),
            post_global=self._escalated_by_question.get(canonical, False),
        )
        _agentic_log(f"post-agent verdict={verdict} info={vinfo}")

        # De-duplicate captured nodes
        seen = set()
        unique_nodes = []
        for n in captured:
            key = (
                n.node.metadata.get("source_path", "") if hasattr(n, "node")
                else n.metadata.get("source_path", ""),
                n.node.text[:80] if hasattr(n, "node") else (n.text[:80] if hasattr(n, "text") else ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_nodes.append(n)

        # Design C: ONE synthesis pass over the merged post-processed nodes,
        # performed by the synthesis model (router never writes the answer).
        if unique_nodes:
            answer = await self.synthesize_with_nodes(question, unique_nodes)
            # Second final refusal: the context was retrieved but the
            # synthesis came back empty/without content.
            if _empty_answer(answer):
                _agentic_log("synthesis returned empty answer; final refusal")
                answer = REFUSAL_FR
        elif agent_failed:
            # No answer at all -> no answer (agentic failure, no rescue).
            answer = REFUSAL_FR
        else:
            # First agentic refusal: the router answered directly with NO tool
            # call (out-of-scope question, e.g. météo/politique/UA-10+).
            # Keep its refusal verbatim ONLY if it genuinely looks like a
            # refusal — otherwise (garbage, truncated reasoning, empty) fall
            # back to the canonical REFUSAL_FR.
            cleaned = (agent_answer or "").strip()
            if cleaned.lower().startswith("answer:"):
                cleaned = cleaned[len("Answer:"):].strip()
            if not cleaned or cleaned.lower() in ("fin", "answer: fin"):
                cleaned = ""
            if cleaned and _is_plausible_refusal(cleaned):
                answer = cleaned
            else:
                answer = REFUSAL_FR
        _agentic_log(f"final answer nodes={len(unique_nodes)}")

        iterations = max(1, len(tools_called))

        # Distinct UA IDs queried, in call order (query_global -> GLOBAL)
        units_called: list[str] = []
        for name, kw in zip(tools_called, tool_kwargs_list):
            ids = (kw or {}).get("ua_ids", [])
            if isinstance(ids, str):
                ids = [ids]
            for u in ids or []:
                if u not in units_called:
                    units_called.append(u)

        return AgenticResult(
            answer=answer,
            tools_called=list(tools_called),
            tool_kwargs=tool_kwargs_list,
            units_called=units_called,
            retrieved_nodes=unique_nodes,
            tokens_in=self.token_counter.prompt_llm_token_count,
            tokens_out=self.token_counter.completion_llm_token_count,
            iterations=iterations,
            model_name=str(self._llm_name),
            embed_model=str(self._embed_name),
            ttft_s=None,
        )

    # ------------------------------------------------------------------
    # Backward-compatible synthesize_with_nodes (used by eval judge_only)
    # ------------------------------------------------------------------
    async def synthesize_with_nodes(
        self,
        question: str,
        nodes: list,
        synth_model: str | None = None,
    ) -> str:
        if not nodes:
            return "Je ne dispose pas de cette information dans la base Activiity."
        synth = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=self._qa_tmpl,
            refine_template=self._refine_tmpl,
            llm=(self.synth_llm if synth_model is None
                 else build_synth_llm(synth_model_override=synth_model)),
        )
        response = await synth.asynthesize(query=question, nodes=nodes)
        return str(response)

    # ------------------------------------------------------------------
    # Sync convenience
    # ------------------------------------------------------------------
    def ask(self, question: str) -> AgenticResult:
        import asyncio
        return asyncio.run(self.achat_with_trace(question))
