"""NaiveRagService — the public interface expected by the eval runner.

Contract (mirrors AgenticRagService):
    async def achat_with_trace(question: str) -> Result
    Result has:
      .answer : str
      .tools_called : list[str]
      .retrieved_nodes : list[NodeWithScore]  (each with .metadata + .text)
      .tokens_in : int
      .tokens_out : int
      .iterations : int
      .model_name : str
      .embed_model : str
      .ttft_s : float | None

The "predicted UA" is inferred by the eval runner from the first
UA-scoped tool name in `tools_called`. To let the naive baseline participate
in UA-routing accuracy measurement *without* actually routing, we pick the
dominant `ua_id` from the top-k retrieved chunks and report it as
`query_ua_X` in `tools_called`. This is post-hoc classification by metadata
vote — NOT an agent decision. The Agentic RAG should beat this.
"""
from __future__ import annotations
import time
from collections import Counter
from dataclasses import dataclass

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Settings as LISettings

from baseline.config import CFG
from baseline.index import build_or_load


@dataclass
class NaiveResult:
    answer: str
    tools_called: list[str]
    retrieved_nodes: list                       # NodeWithScore
    tokens_in: int
    tokens_out: int
    iterations: int
    model_name: str
    embed_model: str
    ttft_s: float | None


class NaiveRagService:
    """One flat index, top-k retrieval, CompactAndRefine synthesis."""

    def __init__(self, data_dir=None, persist_dir=None):
        self.index = build_or_load(
            data_dir=data_dir if data_dir else None or __import__("baseline.config", fromlist=["DATA_DIR"]).DATA_DIR,
            persist_dir=persist_dir if persist_dir else None or __import__("baseline.config", fromlist=["PERSIST_DIR"]).PERSIST_DIR,
        )

        # Token accounting — wired on the global Settings so both retrieval
        # and synthesis get counted.
        self.token_counter = TokenCountingHandler()
        LISettings.callback_manager = CallbackManager([self.token_counter])

        # Single QA prompt (French, grounded, explicit refusal clause).
        qa_tmpl = PromptTemplate(CFG.qa_prompt_fr)
        self.qa_tmpl = qa_tmpl
        self._rebuild_synth()

        self.retriever = self.index.as_retriever(similarity_top_k=CFG.top_k)

        # Cache model names for the trace.
        self._llm_name  = getattr(LISettings.llm, "model", "unknown")
        self._embed_name = getattr(
            LISettings.embed_model,
            "model_name",
            getattr(LISettings.embed_model, "model", "unknown"),
        )

    def _rebuild_synth(self):
        """Recreate the response synthesizer with the current global LLM."""
        self.synth = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=self.qa_tmpl,
            llm=LISettings.llm,
        )

    # ------------------------------------------------------------------
    # Predicted UA via metadata vote — NOT routing, just reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _dominant_ua(nodes) -> str:
        if not nodes:
            return "GLOBAL"
        c: Counter = Counter()
        for n in nodes:
            ua = n.metadata.get("ua_id") if hasattr(n, "metadata") else n.node.metadata.get("ua_id")
            if ua:
                # weighted by similarity score when available
                score = getattr(n, "score", None) or 1.0
                c[ua] += score
        return c.most_common(1)[0][0] if c else "GLOBAL"

    @staticmethod
    def _tool_name_for(ua: str) -> str:
        if ua == "GLOBAL" or not ua:
            return "query_global"
        return f"query_{ua.lower().replace('-', '_')}"

    # ------------------------------------------------------------------
    # Public API (mirror of AgenticRagService.achat_with_trace)
    # ------------------------------------------------------------------
    async def achat_with_trace(self, question: str) -> NaiveResult:
        self.token_counter.reset_counts()
        t0 = time.perf_counter()

        # 1. retrieve — use async to avoid blocking the FastAPI event loop
        nodes = await self.retriever.aretrieve(question)
        t_retrieved = time.perf_counter()

        # 2. synthesize (single LLM call) — async version
        response = await self.synth.asynthesize(query=question, nodes=nodes)
        t_done = time.perf_counter()

        # 3. infer a "tool" name post-hoc from the dominant UA in top-k.
        #    The eval runner parses this to compute routing accuracy.
        dominant = self._dominant_ua(nodes)
        tool_name = self._tool_name_for(dominant)

        answer = str(response)
        tokens_in = self.token_counter.prompt_llm_token_count
        tokens_out = self.token_counter.completion_llm_token_count

        return NaiveResult(
            answer=answer,
            tools_called=[tool_name],
            retrieved_nodes=nodes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            iterations=1,
            model_name=str(self._llm_name),
            embed_model=str(self._embed_name),
            # TTFT is approximated here as time-to-retrieval for the naive RAG.
            # With compact synthesis we don't actually stream; runner accepts None.
            ttft_s=(t_retrieved - t0),
        )

    # Sync convenience for the CLI
    def ask(self, question: str) -> NaiveResult:
        import asyncio
        return asyncio.run(self.achat_with_trace(question))