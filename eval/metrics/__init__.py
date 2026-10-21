"""Metric aggregator — assembles a TierReport from the individual metric modules.

Three layers of metrics:
  1. Judge-free, mandatory (every run):
       - routing (Accuracy/Precision/Recall/F1)
       - refusal (Precision/Recall/F1)
       - retrieval (Precision@k, Recall@k, MRR, nDCG, citation recall)
       - system (latency, tokens, cost)
  2. Judge-free answer quality (recommended, every run when --similarity):
       - EM, token-F1, ROUGE-L, SBERT cosine, BERTScore
  3. Judge-based RAG quality (optional, nightly, LLM judge required):
       - Ragas faithfulness, answer relevancy, context precision/recall
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

from eval.parser import Pair
from eval.runner import Trace
from eval.metrics.classification import routing_report, RoutingReport
from eval.metrics.retrieval import retrieval_report, RetrievalReport
from eval.metrics.refusal import refusal_report, RefusalReport
from eval.metrics.system import system_report, SystemReport


@dataclass
class TierReport:
    tier: str
    n_pairs: int
    routing: RoutingReport
    retrieval: RetrievalReport
    refusal: RefusalReport
    system: SystemReport
    similarity: dict | None = None   # judge-free answer quality (EM, F1, ROUGE, BERTScore)
    ragas: dict | None = None        # judge-based RAG quality (optional, expensive)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "n_pairs": self.n_pairs,
            "routing": asdict(self.routing),
            "retrieval": asdict(self.retrieval),
            "refusal": asdict(self.refusal),
            "system": asdict(self.system),
            "similarity": self.similarity,
            "ragas": self.ragas,
        }


def compute_tier(tier: str, pairs: Iterable[Pair],
                 traces: Iterable[Trace],
                 with_similarity: bool = True,
                 with_ragas: bool = False,
                 use_bertscore: bool = True,
                 use_sbert: bool = True) -> TierReport:
    pairs = list(pairs)
    traces = list(traces)

    rep = TierReport(
        tier=tier,
        n_pairs=len(pairs),
        routing=routing_report(pairs, traces),
        retrieval=retrieval_report(pairs, traces),
        refusal=refusal_report(pairs, traces),
        system=system_report(traces),
    )

    if with_similarity:
        from eval.metrics.similarity import similarity_report
        s = similarity_report(pairs, traces, use_bertscore=use_bertscore, use_sbert=use_sbert)
        rep.similarity = asdict(s)

    if with_ragas:
        from eval.metrics.ragas_bridge import ragas_report
        r = ragas_report(pairs, traces)
        rep.ragas = asdict(r)

    return rep
