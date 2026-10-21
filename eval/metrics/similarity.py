"""Judge-free answer-quality metrics.

These compare the agent's `answer` to the golden `ground_truth` using
deterministic text and embedding similarity — NO generative LLM judge.

Provided metrics:
  - Exact Match (EM)            — strict equality after normalisation
  - Token-F1                    — token-level precision/recall/F1 (SQuAD-style)
  - ROUGE-L F1                  — longest common subsequence overlap
  - BERTScore F1                — embedding-based token similarity (multilingual)
  - SBERT cosine similarity     — sentence-level embedding similarity

All metrics are language-agnostic but we pin multilingual models for French.
Reproducible: same input → same output (no sampling, no temperature).

Why these and not only Ragas?
  - deterministic & fast (no LLM judge, free)
  - capture ~70 % of hallucination/relevance signal
  - Ragas (judge-based) becomes a nightly check rather than a per-PR cost
"""
from __future__ import annotations
import re, string, unicodedata
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean
from typing import Iterable

from eval.parser import Pair
from eval.runner import Trace


# ---------------------------------------------------------------------------
# Normalisation & tokenisation
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation + "«»…“”‘’—–")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))

_STOP_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "à", "a",
    "et", "ou", "est", "sont", "être", "avoir", "ce", "cet", "cette", "ces",
    "il", "elle", "ils", "elles", "on", "nous", "vous", "pour", "par", "sur",
    "avec", "sans", "dans", "en", "que", "qui", "quoi", "dont", "où",
    "mais", "donc", "car", "si", "au", "aux",
}

def normalise(text: str) -> str:
    t = _strip_accents(text.lower())
    t = "".join(" " if c in _PUNCT else c for c in t)
    return re.sub(r"\s+", " ", t).strip()

def tokenise(text: str, drop_stop: bool = True) -> list[str]:
    toks = normalise(text).split()
    if drop_stop:
        toks = [t for t in toks if t not in _STOP_FR]
    return toks


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(pred: str, truth: str) -> float:
    return 1.0 if normalise(pred) == normalise(truth) else 0.0


# ---------------------------------------------------------------------------
# Token-level F1 (SQuAD-style)
# ---------------------------------------------------------------------------

def token_f1(pred: str, truth: str) -> tuple[float, float, float]:
    """Return (precision, recall, f1) on token overlap."""
    pt, tt = tokenise(pred), tokenise(truth)
    if not pt and not tt:
        return 1.0, 1.0, 1.0
    if not pt or not tt:
        return 0.0, 0.0, 0.0
    common: dict[str, int] = {}
    tt_count: dict[str, int] = {}
    for t in tt:
        tt_count[t] = tt_count.get(t, 0) + 1
    for t in pt:
        if tt_count.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            tt_count[t] -= 1
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0, 0.0, 0.0
    p = n_common / len(pt)
    r = n_common / len(tt)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# ROUGE-L (longest common subsequence)
# ---------------------------------------------------------------------------

def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]

def rouge_l(pred: str, truth: str) -> tuple[float, float, float]:
    """Return (precision, recall, f1) of ROUGE-L."""
    pt, tt = tokenise(pred, drop_stop=False), tokenise(truth, drop_stop=False)
    if not pt or not tt:
        return 0.0, 0.0, 0.0
    lcs = _lcs(pt, tt)
    if lcs == 0:
        return 0.0, 0.0, 0.0
    p = lcs / len(pt)
    r = lcs / len(tt)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# Embedding-based metrics: SBERT cosine & BERTScore
#   Pinned to a multilingual model that works well on French.
#   Downloaded on first use; cached in HF cache.
# ---------------------------------------------------------------------------

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERTSCORE_MODEL = "xlm-roberta-base"   # fast, multilingual


@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer
    from eval.metrics.judge_embeddings import JUDGE_EMBED_DEVICE
    return SentenceTransformer(SBERT_MODEL, device=JUDGE_EMBED_DEVICE)


def sbert_cosine(pred: str, truth: str) -> float:
    from sentence_transformers.util import cos_sim
    if not pred.strip() or not truth.strip():
        return 0.0
    emb = _sbert().encode([pred, truth], normalize_embeddings=True)
    return float(cos_sim(emb[0], emb[1]).item())


def bertscore_f1(preds: list[str], truths: list[str]) -> list[float]:
    """Batch BERTScore F1. Returns per-pair F1 scores."""
    import bert_score
    if not preds:
        return []
    _, _, F = bert_score.score(
        preds, truths, model_type=BERTSCORE_MODEL, lang="fr",
        verbose=False, batch_size=32,
    )
    return F.tolist()


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class SimilarityReport:
    n: int
    exact_match: float
    token_f1: float
    token_precision: float
    token_recall: float
    rouge_l_f1: float
    sbert_cosine: float
    bertscore_f1: float | None    # None if bert_score package not installed


def similarity_report(pairs: Iterable[Pair], traces: Iterable[Trace],
                      use_bertscore: bool = True,
                      use_sbert: bool = True) -> SimilarityReport:
    traces_by_id = {t.pair_id: t for t in traces}

    ems, tps, trs, tf1s, rlf1s, sbert = [], [], [], [], [], []
    preds, truths, pair_ids = [], [], []

    # Keep per-pair values so we can attach them to Trace objects.
    per_pair: dict[str, dict[str, float]] = {}
    for p in pairs:
        t = traces_by_id.get(p.id)
        if t is None or t.error or not p.ground_truth or not t.answer:
            continue
        em = exact_match(t.answer, p.ground_truth)
        tp, tr, tf1 = token_f1(t.answer, p.ground_truth)
        _, _, rf1 = rouge_l(t.answer, p.ground_truth)

        ems.append(em)
        tps.append(tp)
        trs.append(tr)
        tf1s.append(tf1)
        rlf1s.append(rf1)

        preds.append(t.answer)
        truths.append(p.ground_truth)
        pair_ids.append(p.id)

        per_pair[p.id] = {
            "exact_match": float(em),
            "token_precision": float(tp),
            "token_recall": float(tr),
            "token_f1": float(tf1),
            "rouge_l_f1": float(rf1),
        }

    sbert_per_pair: list[float] | None = None
    if use_sbert and preds:
        try:
            sbert_per_pair = [sbert_cosine(a, b) for a, b in zip(preds, truths)]
            sbert = list(sbert_per_pair)
        except ImportError:
            sbert_per_pair = None
            sbert = []

    bert_f1 = None
    bert_f1_per_pair: list[float] | None = None
    if use_bertscore and preds:
        try:
            bert_f1_per_pair = bertscore_f1(preds, truths)
            bert_f1 = mean(bert_f1_per_pair) if bert_f1_per_pair else None
        except ImportError:
            bert_f1_per_pair = None
            bert_f1 = None

    # Attach per-pair similarity onto Trace objects (like Ragas does).
    for i, pid in enumerate(pair_ids):
        t = traces_by_id.get(pid)
        if t is None:
            continue
        block = per_pair.get(pid, {}).copy()
        block["sbert_cosine"] = float(sbert_per_pair[i]) if sbert_per_pair is not None else None
        block["bertscore_f1"] = float(bert_f1_per_pair[i]) if bert_f1_per_pair is not None else None
        t.similarity = block
        t.similarity_error = None

    n = len(ems)
    if n == 0:
        return SimilarityReport(0, 0, 0, 0, 0, 0, 0, None)

    return SimilarityReport(
        n=n,
        exact_match=mean(ems),
        token_f1=mean(tf1s),
        token_precision=mean(tps),
        token_recall=mean(trs),
        rouge_l_f1=mean(rlf1s),
        sbert_cosine=mean(sbert) if sbert else 0.0,
        bertscore_f1=bert_f1,
    )
