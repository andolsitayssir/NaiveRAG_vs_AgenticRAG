"""Unit tests for eval.metrics.retrieval — pure functions, no LLM."""
import pytest
from eval.metrics.retrieval import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, citation_recall,
)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_perfect():
    assert precision_at_k(["a.docx", "b.docx"], ["a.docx"], k=1) == 1.0

def test_precision_partial():
    # 1 of 2 top-k chunks relevant
    assert precision_at_k(["a.docx", "c.pdf"], ["a.docx"], k=2) == 0.5

def test_precision_empty_retrieved():
    assert precision_at_k([], ["a.docx"], k=5) == 0.0

def test_precision_no_must_cite_is_perfect():
    # no must_cite => trivially perfect (used for refusal pairs)
    assert precision_at_k(["whatever.pdf"], [], k=5) == 1.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_all_found():
    assert recall_at_k(["a.docx", "b.pdf"], ["a.docx", "b.pdf"], k=5) == 1.0

def test_recall_half_found():
    assert recall_at_k(["a.docx"], ["a.docx", "b.pdf"], k=5) == 0.5

def test_recall_basename_match():
    # even with full path, basename match counts
    assert recall_at_k(["/x/y/a.docx"], ["a.docx"], k=5) == 1.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

def test_mrr_first_rank():
    assert mrr(["a.docx", "x"], ["a.docx"]) == 1.0

def test_mrr_second_rank():
    assert mrr(["x", "a.docx"], ["a.docx"]) == pytest.approx(0.5)

def test_mrr_not_found():
    assert mrr(["x", "y"], ["a.docx"]) == 0.0


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------

def test_ndcg_ideal():
    # if the only relevant doc is first, nDCG = 1.0
    assert ndcg_at_k(["a.docx", "x", "y"], ["a.docx"], k=3) == pytest.approx(1.0)

def test_ndcg_second_position_penalty():
    v = ndcg_at_k(["x", "a.docx"], ["a.docx"], k=3)
    assert 0 < v < 1


# ---------------------------------------------------------------------------
# citation_recall (binary)
# ---------------------------------------------------------------------------

def test_citation_recall_hit():
    assert citation_recall(["a.docx"], ["a.docx"]) == 1.0

def test_citation_recall_miss():
    assert citation_recall(["b.pdf"], ["a.docx"]) == 0.0
