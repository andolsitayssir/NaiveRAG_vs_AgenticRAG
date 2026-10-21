"""Bridge to Ragas for generative quality metrics.

Ragas metrics used:
  - faithfulness        : answer sticks to retrieved context (anti-hallucination)
  - answer_relevancy    : answer is responsive to the question
  - context_precision   : retrieved chunks are useful for the answer
  - context_recall      : retrieved chunks cover what the ground truth needed
  - answer_correctness  : RN ≈ GT (claim-level F1 with bidirectional entailment)

Ragas calls an LLM judge — we pin the judge (config.JUDGE) for reproducibility.

Scoring scope: only pairs that actually retrieved context(s) enter the Ragas
dataset. Out-of-scope / refused pairs retrieve nothing and are unscorable —
they stay 0.0 in the routing/refusal/retrieval tiers instead of dragging the
generative averages down as fake zeros.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from statistics import mean
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset

from eval.config import JUDGE
from eval.runner import Trace
from eval.parser import Pair


@dataclass
class RagasReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float   # the "RN ≈ GT" headline accuracy
    n_samples: int


def _as_dataset(pairs: list[Pair], traces: dict[str, Trace]) -> "Dataset":
    from datasets import Dataset
    rows = []
    for p in pairs:
        t = traces.get(p.id)
        # RAGAS needs retrieved contexts + a real answer to score anything.
        # Pairs that retrieved nothing (correct OOS refusals, and refused
        # answers with no context) are unscorable: they would be averaged in
        # as 0.0 on every metric, dragging the tier down without saying
        # anything about generative quality. They are already scored by the
        # routing / refusal / retrieval tiers, so skip them here (consistently
        # for the primary path AND the ordering-based fallback below).
        if t is None or t.error or not p.ground_truth:
            continue
        if not (t.retrieved_chunks or []):
            continue
        rows.append({
            "pair_id": p.id,
            "question": p.question,
            "answer": t.answer,
            "contexts": t.retrieved_chunks,
            "ground_truth": p.ground_truth,
        })
    return Dataset.from_list(rows)


def _build_judge(api_key: str | None = None):
    """Return a Ragas-compatible LLM (LangchainLLMWrapper) for the judge.

    Ragas 0.4+ uses LangChain chat models rather than LlamaIndex LLMs.
    We route OpenRouter traffic through ChatOpenAI with base_url set.
    
    api_key can be user-provided or the server's configured key."""
    from ragas.llms import LangchainLLMWrapper

    if not api_key:
        raise ValueError("OpenRouter API key is required.")

    if JUDGE.provider == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=JUDGE.model,
            temperature=JUDGE.temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60,
            max_retries=JUDGE.max_retries,
            max_tokens=2048,  # Request fewer tokens to avoid credit limit errors
        )
        return LangchainLLMWrapper(llm)

    if JUDGE.provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=JUDGE.model,
            temperature=JUDGE.temperature,
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
        return LangchainLLMWrapper(llm)

    if JUDGE.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=JUDGE.model,
            temperature=JUDGE.temperature,
            google_api_key=api_key,
        )
        return LangchainLLMWrapper(llm)

    raise ValueError(f"Unknown judge provider: {JUDGE.provider}")


def _build_judge_embeddings():
    """Some Ragas metrics (answer_relevancy, answer_correctness) need embeddings
    in addition to the judge LLM. Use local sentence-transformers."""
    from eval.metrics.judge_embeddings import build_judge_embeddings
    return build_judge_embeddings()


def ragas_report(pairs: Iterable[Pair], traces: Iterable[Trace]) -> RagasReport:
    pairs = list(pairs)
    traces_by_id = {t.pair_id: t for t in traces}

    ds = _as_dataset(pairs, traces_by_id)
    if len(ds) == 0:
        return RagasReport(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    from ragas import evaluate
    # Note: Ragas metrics are objects in 0.4+.
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    judge = _build_judge(os.getenv("OPENROUTER_API_KEY"))
    emb = _build_judge_embeddings()

    m_faith = Faithfulness(llm=judge)
    # Ragas defaults can request multiple generations (strictness=3).
    # Some providers/models ignore n>1 and emit warnings; strictness=1 keeps
    # the metric deterministic and typically faster.
    try:
        m_relevancy = AnswerRelevancy(llm=judge, embeddings=emb, strictness=1)
    except TypeError:
        m_relevancy = AnswerRelevancy(llm=judge, embeddings=emb)
    m_precision = ContextPrecision(llm=judge)
    m_recall = ContextRecall(llm=judge)
    try:
        m_correctness = AnswerCorrectness(llm=judge, embeddings=emb, strictness=1)
    except TypeError:
        m_correctness = AnswerCorrectness(llm=judge, embeddings=emb)

    # Force n=1 and use RunConfig to stay under 10 RPM limit
    for m in [m_faith, m_relevancy, m_precision, m_recall, m_correctness]:
        if hasattr(m, "n"):
            m.n = 1
    
    from ragas.run_config import RunConfig
    max_workers = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
    timeout = int(os.getenv("RAGAS_TIMEOUT_S", "120"))
    results = evaluate(
        ds,
        metrics=[m_faith, m_relevancy, m_precision, m_recall, m_correctness],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=max_workers, timeout=timeout)
    )
    df = results.to_pandas()

    # ------------------------------------------------------------------
    # Attach per-pair scores back into the Trace objects
    # ------------------------------------------------------------------
    def _val(row, col: str):
        v = row.get(col, None)
        try:
            # pandas may give NaN; NaN != NaN is a portable check
            if v is None:
                return None
            if isinstance(v, float) and v != v:
                return None
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    if "pair_id" in df.columns:
        for _, row in df.iterrows():
            pid = row.get("pair_id")
            if not pid:
                continue
            t = traces_by_id.get(str(pid))
            if t is None:
                continue
            t.ragas = {
                "faithfulness": _val(row, "faithfulness"),
                "answer_relevancy": _val(row, "answer_relevancy"),
                "context_precision": _val(row, "context_precision"),
                "context_recall": _val(row, "context_recall"),
                "answer_correctness": _val(row, "answer_correctness"),
            }
            t.ragas_error = None
    else:
        # Fallback: keep ordering-based mapping if Ragas drops custom columns.
        # ds preserves row order; map back to the subset of eligible pair_ids.
        eligible = [
            p.id for p in pairs
            if (p.id in traces_by_id
                and not traces_by_id[p.id].error
                and p.ground_truth
                and (traces_by_id[p.id].retrieved_chunks or []))
        ]
        for i, pid in enumerate(eligible[: len(df)]):
            t = traces_by_id.get(pid)
            if t is None:
                continue
            row = df.iloc[i].to_dict()
            t.ragas = {
                "faithfulness": _val(row, "faithfulness"),
                "answer_relevancy": _val(row, "answer_relevancy"),
                "context_precision": _val(row, "context_precision"),
                "context_recall": _val(row, "context_recall"),
                "answer_correctness": _val(row, "answer_correctness"),
            }
            t.ragas_error = None

    def _avg(col: str) -> float:
        series = df[col].dropna()
        return float(series.mean()) if len(series) else 0.0

    return RagasReport(
        faithfulness=_avg("faithfulness"),
        answer_relevancy=_avg("answer_relevancy"),
        context_precision=_avg("context_precision"),
        context_recall=_avg("context_recall"),
        answer_correctness=_avg("answer_correctness"),
        n_samples=len(ds),
    )
