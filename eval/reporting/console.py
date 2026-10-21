"""Pretty console summary for a TierReport."""
from __future__ import annotations
import sys
from eval.metrics import TierReport
from eval.config import THRESHOLDS

# Windows console is often cp1252 — force UTF-8 if possible.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def _pass(value: float, threshold: float, higher_is_better: bool = True) -> str:
    ok = value >= threshold if higher_is_better else value <= threshold
    return "PASS" if ok else "FAIL"


def print_tier(r: TierReport):
    th = THRESHOLDS[r.tier]
    print(f"\n━━━━━━━━━━━━━━━━━━━━  TIER: {r.tier.upper()}  ({r.n_pairs} pairs)  ━━━━━━━━━━━━━━━━━━━━\n")

    print("  ROUTING (UA classification)")
    print(f"    accuracy         : {r.routing.accuracy:.3f}   threshold {th.accuracy:.2f}  [{_pass(r.routing.accuracy, th.accuracy)}]")
    print(f"    macro-F1         : {r.routing.macro_f1:.3f}   threshold {th.macro_f1:.2f}  [{_pass(r.routing.macro_f1, th.macro_f1)}]")
    print(f"    macro-precision  : {r.routing.macro_precision:.3f}")
    print(f"    macro-recall     : {r.routing.macro_recall:.3f}")
    print(f"    weighted-F1      : {r.routing.weighted_f1:.3f}")

    print("\n  REFUSAL (out-of-scope)")
    print(f"    precision        : {r.refusal.precision:.3f}   threshold {th.refusal_precision:.2f}  [{_pass(r.refusal.precision, th.refusal_precision)}]")
    print(f"    recall           : {r.refusal.recall:.3f}   threshold {th.refusal_recall:.2f}  [{_pass(r.refusal.recall, th.refusal_recall)}]")
    print(f"    F1               : {r.refusal.f1:.3f}")
    print(f"    TP/FP/TN/FN      : {r.refusal.true_positives}/{r.refusal.false_positives}/{r.refusal.true_negatives}/{r.refusal.false_negatives}")
    print(f"    borderline OK    : {r.refusal.borderline_correct}/{r.refusal.borderline_total}")

    print("\n  RETRIEVAL")
    print(f"    precision@{r.retrieval.k}      : {r.retrieval.precision_at_k:.3f}")
    print(f"    recall@{r.retrieval.k}         : {r.retrieval.recall_at_k:.3f}")
    print(f"    MRR              : {r.retrieval.mrr:.3f}")
    print(f"    nDCG@{r.retrieval.k}           : {r.retrieval.ndcg_at_k:.3f}")
    print(f"    citation recall  : {r.retrieval.citation_recall:.3f}   threshold {th.citation_recall:.2f}  [{_pass(r.retrieval.citation_recall, th.citation_recall)}]")

    if r.similarity:
        s = r.similarity
        print("\n  ANSWER SIMILARITY (judge-free)")
        print(f"    exact match      : {s['exact_match']:.3f}")
        print(f"    token-F1         : {s['token_f1']:.3f}   threshold {th.token_f1:.2f}  [{_pass(s['token_f1'], th.token_f1)}]")
        print(f"    token P / R      : {s['token_precision']:.3f} / {s['token_recall']:.3f}")
        print(f"    ROUGE-L F1       : {s['rouge_l_f1']:.3f}   threshold {th.rouge_l_f1:.2f}  [{_pass(s['rouge_l_f1'], th.rouge_l_f1)}]")
        if s.get("sbert_cosine"):
            print(f"    SBERT cosine     : {s['sbert_cosine']:.3f}   threshold {th.sbert_cosine:.2f}  [{_pass(s['sbert_cosine'], th.sbert_cosine)}]")
        if s.get("bertscore_f1") is not None:
            print(f"    BERTScore F1     : {s['bertscore_f1']:.3f}   threshold {th.bertscore_f1:.2f}  [{_pass(s['bertscore_f1'], th.bertscore_f1)}]")

    if r.ragas:
        rg = r.ragas
        print("\n  RAG QUALITY (Ragas — judge-based)")
        if "answer_correctness" in rg:
            print(f"    answer_correctness : {rg['answer_correctness']:.3f}   ← RN ≈ GT headline accuracy")
        print(f"    faithfulness     : {rg['faithfulness']:.3f}   threshold {th.faithfulness:.2f}  [{_pass(rg['faithfulness'], th.faithfulness)}]")
        print(f"    answer_relevancy : {rg['answer_relevancy']:.3f}   threshold {th.answer_relevancy:.2f}  [{_pass(rg['answer_relevancy'], th.answer_relevancy)}]")
        print(f"    context_precision: {rg['context_precision']:.3f}   threshold {th.context_precision:.2f}  [{_pass(rg['context_precision'], th.context_precision)}]")
        print(f"    context_recall   : {rg['context_recall']:.3f}   threshold {th.context_recall:.2f}  [{_pass(rg['context_recall'], th.context_recall)}]")

    print("\n  SYSTEM")
    print(f"    errors           : {r.system.errors}/{r.system.n}")
    print(f"    latency p50/p95  : {r.system.latency_p50_s:.2f}s / {r.system.latency_p95_s:.2f}s   p95 threshold {th.p95_latency_s:.1f}s  [{_pass(r.system.latency_p95_s, th.p95_latency_s, higher_is_better=False)}]")
    print(f"    TTFT p50         : {r.system.ttft_p50_s:.2f}s")
    print(f"    avg tokens in/out: {r.system.avg_tokens_in:.0f} / {r.system.avg_tokens_out:.0f}")
    print(f"    total cost       : ${r.system.cost_usd:.4f}")
    print(f"    iterations>4     : {r.system.over_cap_iterations} (must be 0)")
