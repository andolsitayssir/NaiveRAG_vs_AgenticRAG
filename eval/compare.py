"""Compare two eval runs and fail if any metric regresses by more than N points.

Usage:
    python -m eval.compare reports/baseline reports/candidate
    python -m eval.compare reports/baseline reports/candidate --fail-on-regress 2
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from eval.config import MAX_REGRESS_POINTS


# metric_path -> (higher_is_better,)
TRACKED: dict[str, bool] = {
    "routing.accuracy": True,
    "routing.macro_f1": True,
    "routing.macro_precision": True,
    "routing.macro_recall": True,
    "routing.weighted_f1": True,
    "refusal.precision": True,
    "refusal.recall": True,
    "refusal.f1": True,
    "retrieval.precision_at_k": True,
    "retrieval.recall_at_k": True,
    "retrieval.mrr": True,
    "retrieval.ndcg_at_k": True,
    "retrieval.citation_recall": True,
    "similarity.exact_match": True,
    "similarity.token_f1": True,
    "similarity.rouge_l_f1": True,
    "similarity.sbert_cosine": True,
    "similarity.bertscore_f1": True,
    "ragas.faithfulness": True,
    "ragas.answer_relevancy": True,
    "ragas.context_precision": True,
    "ragas.context_recall": True,
    "system.latency_p95_s": False,
    "system.latency_p50_s": False,
    "system.cost_usd": False,
}


def _get(d: dict, path: str):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def compare(baseline: dict, candidate: dict, max_pts: float) -> list[str]:
    failures = []
    for tier in candidate:
        for metric, higher in TRACKED.items():
            b = _get(baseline.get(tier, {}), metric)
            c = _get(candidate.get(tier, {}), metric)
            if b is None or c is None:
                continue
            delta = (c - b) * 100   # percentage points for ratios, seconds*100 for latency
            # For latency & cost, "worse" is higher; invert the sign for uniform reasoning
            regress = -delta if higher else delta
            if regress > max_pts:
                arrow = "↓" if higher else "↑"
                failures.append(
                    f"[{tier}] {metric}: {b:.3f} → {c:.3f} ({arrow}{regress:.1f} pts)"
                )
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--fail-on-regress", type=float, default=MAX_REGRESS_POINTS,
                    help="Fail if any metric drops by more than this many points")
    args = ap.parse_args()

    b = json.loads((args.baseline / "report.json").read_text(encoding="utf-8"))
    c = json.loads((args.candidate / "report.json").read_text(encoding="utf-8"))

    failures = compare(b, c, args.fail_on_regress)
    if failures:
        print("REGRESSION DETECTED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"OK: no metric regressed by more than {args.fail_on_regress} points.")


if __name__ == "__main__":
    main()
