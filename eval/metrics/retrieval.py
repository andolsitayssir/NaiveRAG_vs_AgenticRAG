"""Retrieval metrics: Precision@k, Recall@k, MRR, nDCG@k, Citation Recall."""
from __future__ import annotations
from dataclasses import dataclass
from math import log2
from statistics import mean
from typing import Iterable

from eval.config import RETRIEVAL_K
from eval.parser import Pair
from eval.runner import Trace


def _relevant(retrieved_path: str, must_cite: list[str]) -> bool:
    """A retrieved chunk is relevant if any of its must_cite patterns matches
    its source_path (basename match)."""
    from pathlib import PurePath
    base = PurePath(retrieved_path).name
    return any(PurePath(mc).name == base or mc == retrieved_path
               or PurePath(mc).name in base
               for mc in must_cite)


def precision_at_k(retrieved: list[str], must_cite: list[str], k: int) -> float:
    if not must_cite:
        return 1.0   # nothing to retrieve -> trivially perfect
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(_relevant(r, must_cite) for r in top)
    return hits / len(top)


def recall_at_k(retrieved: list[str], must_cite: list[str], k: int) -> float:
    if not must_cite:
        return 1.0
    top = retrieved[:k]
    # multiset of distinct must_cite items found
    found = {mc for mc in must_cite
             if any(_relevant(r, [mc]) for r in top)}
    return len(found) / len(must_cite)


def mrr(retrieved: list[str], must_cite: list[str]) -> float:
    if not must_cite:
        return 1.0
    for i, r in enumerate(retrieved, start=1):
        if _relevant(r, must_cite):
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], must_cite: list[str], k: int) -> float:
    if not must_cite:
        return 1.0
    gains = [1.0 if _relevant(r, must_cite) else 0.0 for r in retrieved[:k]]
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(must_cite), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def citation_recall(retrieved: list[str], must_cite: list[str]) -> float:
    """1 if any of the must_cite files is present in the retrieved sources,
    else 0 (binary, as used in the design-doc release gate)."""
    if not must_cite:
        return 1.0
    return 1.0 if any(_relevant(r, must_cite) for r in retrieved) else 0.0


@dataclass
class RetrievalReport:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    citation_recall: float
    k: int


def retrieval_report(pairs: Iterable[Pair], traces: Iterable[Trace],
                     k: int = RETRIEVAL_K) -> RetrievalReport:
    pairs_by_id = {p.id: p for p in pairs}
    ps, rs, mrrs, ndcgs, cits = [], [], [], [], []
    for t in traces:
        p = pairs_by_id.get(t.pair_id)
        if p is None or not p.must_cite:
            continue  # skip refusal-only pairs from retrieval stats
        ps.append(precision_at_k(t.retrieved_sources, p.must_cite, k))
        rs.append(recall_at_k(t.retrieved_sources, p.must_cite, k))
        mrrs.append(mrr(t.retrieved_sources, p.must_cite))
        ndcgs.append(ndcg_at_k(t.retrieved_sources, p.must_cite, k))
        cits.append(citation_recall(t.retrieved_sources, p.must_cite))
    if not ps:
        return RetrievalReport(0.0, 0.0, 0.0, 0.0, 0.0, k)
    return RetrievalReport(
        precision_at_k=mean(ps),
        recall_at_k=mean(rs),
        mrr=mean(mrrs),
        ndcg_at_k=mean(ndcgs),
        citation_recall=mean(cits),
        k=k,
    )
