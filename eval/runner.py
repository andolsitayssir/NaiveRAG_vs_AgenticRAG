"""Drive the Agentic RAG service on each Pair and collect a Trace."""
from __future__ import annotations
import asyncio, os, re, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from eval.config import REFUSAL_PATTERNS
from eval.parser import Pair


@dataclass
class Trace:
    """Everything we record for one Q&A pair execution."""
    pair_id: str
    tier: str
    bucket: str

    # agent decisions
    tools_called: list[str] = field(default_factory=list)
    tool_kwargs: list[dict] = field(default_factory=list)   # NEW: kwargs per tool call
    predicted_ua: str | None = None      # derived from the first UA-scoped tool call
    is_refusal: bool = False

    # retrieval
    retrieved_sources: list[str] = field(default_factory=list)  # ordered list of source_path
    retrieved_chunks: list[str] = field(default_factory=list)   # chunk texts for Ragas

    # generation
    answer: str = ""
    ground_truth: str = ""
    question: str = ""

    # system
    latency_s: float = 0.0
    ttft_s: float | None = None          # time to first token
    tokens_in: int = 0
    tokens_out: int = 0
    model_name: str = ""
    embed_model: str = ""
    iterations: int = 0

    # status
    error: str | None = None

    # judge (optional)
    # Per-pair Ragas metrics are attached by eval.metrics.ragas_bridge.
    ragas: dict[str, float | None] | None = None
    ragas_error: str | None = None

    # judge-free similarity (optional)
    # Per-pair values can be attached by eval.metrics.similarity.
    similarity: dict[str, float | None] | None = None
    similarity_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_UA_RE = re.compile(r"UA[_-](\d+)", re.IGNORECASE)


def _infer_predicted_ua(tool_names: list[str], tool_kwargs_list: list[dict] | None = None) -> str | None:
    """Infer the predicted UA from tool calls.

    v4: Reads 'ua_ids' (list) from tool_kwargs — query_by_ua batches all
        relevant UAs in ONE call; we attribute the routing to the first
        listed UA (deterministic; multi-accept truth handled by the metric).
    v3: Falls back to 'ua_id' (str) for the legacy single-UA signature.
    v2: Checks tool name for 'UA-N' pattern (11-tool design).
    If only query_global was called, return 'GLOBAL'. If no tool, return None.
    """
    kwargs_list = tool_kwargs_list or []
    for i, t in enumerate(tool_names):
        # v4: check tool_kwargs for ua_ids (list) parameter
        if i < len(kwargs_list):
            kw = kwargs_list[i]
            ua_ids = kw.get("ua_ids")
            if isinstance(ua_ids, list) and ua_ids:
                m = _UA_RE.search(str(ua_ids[0]))
                if m:
                    return f"UA-{m.group(1)}"
            # v3 fallback: single ua_id string
            ua_id = kw.get("ua_id", "")
            if ua_id:
                m = _UA_RE.search(ua_id)
                if m:
                    return f"UA-{m.group(1)}"
        # v2 fallback: check tool name for UA pattern
        m = _UA_RE.search(t)
        if m:
            return f"UA-{m.group(1)}"
        if "global" in t.lower():
            continue
    # tools were called but none are UA-scoped: it's global or empty
    return "GLOBAL" if any("global" in t.lower() for t in tool_names) else None


def _is_refusal(answer: str) -> bool:
    return any(re.search(p, answer) for p in REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Real runner — calls into the AgenticRagService
# ---------------------------------------------------------------------------

async def run_pair(svc, pair: Pair) -> Trace:
    """Run a single pair against the LlamaIndex agent. `svc` is expected to
    expose `achat_with_trace(question)` that returns a dict-like object with
    `answer`, `tools_called`, `retrieved_nodes`, `tokens_in`, `tokens_out`,
    `iterations`, and emits first-token timing via a callback."""
    trace = Trace(
        pair_id=pair.id, tier=pair.tier, bucket=pair.bucket,
        question=pair.question, ground_truth=pair.ground_truth,
    )
    t0 = time.perf_counter()

    def _node_meta(n) -> dict:
        md = getattr(n, "metadata", None)
        if isinstance(md, dict):
            return md
        node = getattr(n, "node", None)
        md2 = getattr(node, "metadata", None)
        return md2 if isinstance(md2, dict) else {}

    def _node_text(n) -> str:
        txt = getattr(n, "text", None)
        if isinstance(txt, str):
            return txt
        get_content = getattr(n, "get_content", None)
        if callable(get_content):
            try:
                return str(get_content())
            except Exception:  # noqa: BLE001
                pass
        node = getattr(n, "node", None)
        if node is not None:
            txt2 = getattr(node, "text", None)
            if isinstance(txt2, str):
                return txt2
            get_content2 = getattr(node, "get_content", None)
            if callable(get_content2):
                try:
                    return str(get_content2())
                except Exception:  # noqa: BLE001
                    pass
        return ""

    try:
        result = await svc.achat_with_trace(pair.question)
        trace.latency_s = time.perf_counter() - t0
        trace.ttft_s = getattr(result, "ttft_s", None)
        trace.answer = result.answer
        trace.tools_called = list(result.tools_called)
        trace.tool_kwargs = list(getattr(result, "tool_kwargs", []))
        trace.retrieved_sources = [
            _node_meta(n).get("source_path", "") for n in result.retrieved_nodes
        ]
        trace.retrieved_chunks = [_node_text(n) for n in result.retrieved_nodes]
        trace.tokens_in = result.tokens_in
        trace.tokens_out = result.tokens_out
        trace.iterations = result.iterations
        trace.model_name = result.model_name
        trace.embed_model = result.embed_model
    except Exception as e:  # noqa: BLE001
        trace.latency_s = time.perf_counter() - t0
        trace.error = f"{type(e).__name__}: {e}"
    finally:
        trace.predicted_ua = _infer_predicted_ua(trace.tools_called, trace.tool_kwargs)
        trace.is_refusal = _is_refusal(trace.answer)
    return trace


async def run_pairs(svc, pairs: Iterable[Pair], workers: int = 6) -> list[Trace]:
    """Run pairs with a bounded concurrency."""
    pair_list = list(pairs)
    total = len(pair_list)
    if total == 0:
        return []

    sem = asyncio.Semaphore(workers)

    progress_every = int(os.getenv("EVAL_PROGRESS_EVERY", "10"))
    progress_min_s = float(os.getenv("EVAL_PROGRESS_MIN_S", "15"))
    started = time.perf_counter()
    last_print = 0.0

    results: list[Trace | None] = [None] * total
    completed = 0
    errors = 0

    async def _run_one(i: int, p: Pair):
        async with sem:
            return i, await run_pair(svc, p)

    tasks = [asyncio.create_task(_run_one(i, p)) for i, p in enumerate(pair_list)]
    try:
        for fut in asyncio.as_completed(tasks):
            i, trace = await fut
            results[i] = trace
            completed += 1
            if trace.error:
                errors += 1

            now = time.perf_counter()
            should_print = (
                completed == total
                or (completed % max(progress_every, 1) == 0 and (now - last_print) >= progress_min_s)
            )
            if should_print:
                elapsed = now - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"[eval] {completed}/{total} done (errors={errors})  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                    flush=True,
                )
                last_print = now
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()

    # Safety: should be fully populated, but keep type checker happy.
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Mock runner — deterministic fake for smoke tests & CI without LLMs
# ---------------------------------------------------------------------------

class MockSvc:
    """Deterministic mock: returns ground_truth as answer, cites must_cite,
    picks expected_ua as tool. Useful for testing the eval pipeline itself."""

    def __init__(self, pairs_by_qid: dict[str, Pair]):
        self._pairs = pairs_by_qid

    class _Result:
        def __init__(self, **kw): self.__dict__.update(kw)

    async def achat_with_trace(self, question: str):
        # find the pair by question text
        pair = next((p for p in self._pairs.values() if p.question == question), None)
        if pair is None:
            return self._Result(
                answer="Je ne trouve pas cette information dans la base Activiity.",
                tools_called=[],
                tool_kwargs=[],
                retrieved_nodes=[],
                tokens_in=50, tokens_out=20,
                iterations=1, model_name="mock", embed_model="mock",
                ttft_s=0.01,
            )
        exp = pair.expected_uas[0] if pair.expected_uas else "GLOBAL"
        if exp == "NONE":
            return self._Result(
                answer="Je ne trouve pas cette information dans la base Activiity. Pouvez-vous reformuler ?",
                tools_called=[],
                tool_kwargs=[],
                retrieved_nodes=[],
                tokens_in=50, tokens_out=25,
                iterations=1, model_name="mock", embed_model="mock",
                ttft_s=0.01,
            )
        # v4: use query_by_ua with ua_ids (list) in kwargs
        if exp == "GLOBAL":
            tool = "query_global"
            kwargs = {"question": pair.question}
        else:
            tool = "query_by_ua"
            kwargs = {"ua_ids": [exp], "question": pair.question}
        class _Node:
            def __init__(self, path: str, text: str):
                self.metadata = {"source_path": path}
                self.text = text
        nodes = [_Node(p, pair.ground_truth[:200]) for p in (pair.must_cite or ["unknown"])]
        return self._Result(
            answer=pair.ground_truth,
            tools_called=[tool],
            tool_kwargs=[kwargs],
            retrieved_nodes=nodes,
            tokens_in=800, tokens_out=150,
            iterations=1, model_name="mock", embed_model="mock",
            ttft_s=0.05,
        )
