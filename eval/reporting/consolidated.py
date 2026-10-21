"""Consolidated report — unifies deterministic (judge-free) + Ragas (judge-based)
metrics into a single decision-ready view of one run.

The idea: every layer of the evaluator measures something different. Looking at
them in isolation hides correlations. The consolidated report:

  1. shows all 5 metric layers side by side, with pass/fail vs. tier thresholds
  2. computes a weighted "Quality Score" that combines deterministic + judge
     signals (so a single number can be tracked across runs)
  3. lists the top 5 actionable failure modes ranked by impact

It reads `report.json` from a completed run directory and writes:
  - `consolidated.md`  — human-readable digest
  - `consolidated.json` — machine-readable, for CI dashboards
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.config import THRESHOLDS, Threshold


# ---------------------------------------------------------------------------
# Quality Score formula
# ---------------------------------------------------------------------------
# Weighted average over [0, 1] sub-scores. Weights chosen to:
#  - give the judge half the weight when it's available (it's the most reliable
#    proxy for human judgment)
#  - never let one layer single-handedly dominate
#  - degrade gracefully when a layer is missing (re-normalise weights)

WEIGHTS_WITH_JUDGE = {
    "answer_correctness": 0.30,   # the headline RN ≈ GT score
    "faithfulness":       0.10,
    "answer_relevancy":   0.05,
    "macro_f1":           0.15,   # routing
    "citation_recall":    0.10,
    "refusal_f1":         0.10,
    "token_f1":           0.10,   # judge-free answer overlap
    "rouge_l_f1":         0.05,
    "mrr":                0.05,
}

WEIGHTS_NO_JUDGE = {
    "macro_f1":         0.30,
    "citation_recall":  0.20,
    "refusal_f1":       0.20,
    "token_f1":         0.20,
    "rouge_l_f1":       0.05,
    "mrr":              0.05,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path(d: dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default


def _flatten(rep: dict) -> dict[str, float]:
    """Pull every leaf metric we care about into a flat dict."""
    return {
        # routing
        "accuracy":           _path(rep, "routing", "accuracy", default=0.0),
        "macro_precision":    _path(rep, "routing", "macro_precision", default=0.0),
        "macro_recall":       _path(rep, "routing", "macro_recall", default=0.0),
        "macro_f1":           _path(rep, "routing", "macro_f1", default=0.0),
        "weighted_f1":        _path(rep, "routing", "weighted_f1", default=0.0),
        # refusal
        "refusal_precision":  _path(rep, "refusal", "precision", default=0.0),
        "refusal_recall":     _path(rep, "refusal", "recall", default=0.0),
        "refusal_f1":         _path(rep, "refusal", "f1", default=0.0),
        # retrieval
        "precision_at_k":     _path(rep, "retrieval", "precision_at_k", default=0.0),
        "recall_at_k":        _path(rep, "retrieval", "recall_at_k", default=0.0),
        "mrr":                _path(rep, "retrieval", "mrr", default=0.0),
        "ndcg_at_k":          _path(rep, "retrieval", "ndcg_at_k", default=0.0),
        "citation_recall":    _path(rep, "retrieval", "citation_recall", default=0.0),
        # similarity (judge-free)
        "exact_match":        _path(rep, "similarity", "exact_match", default=0.0),
        "token_f1":           _path(rep, "similarity", "token_f1", default=0.0),
        "rouge_l_f1":         _path(rep, "similarity", "rouge_l_f1", default=0.0),
        "sbert_cosine":       _path(rep, "similarity", "sbert_cosine", default=0.0),
        "bertscore_f1":       _path(rep, "similarity", "bertscore_f1") or 0.0,
        # ragas (judge-based)
        "faithfulness":       _path(rep, "ragas", "faithfulness", default=0.0),
        "answer_relevancy":   _path(rep, "ragas", "answer_relevancy", default=0.0),
        "context_precision":  _path(rep, "ragas", "context_precision", default=0.0),
        "context_recall":     _path(rep, "ragas", "context_recall", default=0.0),
        "answer_correctness": _path(rep, "ragas", "answer_correctness", default=0.0),
        # system
        "latency_p50":        _path(rep, "system", "latency_p50_s", default=0.0),
        "latency_p95":        _path(rep, "system", "latency_p95_s", default=0.0),
        "cost_usd":           _path(rep, "system", "cost_usd", default=0.0),
        "errors":             _path(rep, "system", "errors", default=0),
        "n":                  _path(rep, "n_pairs", default=0),
    }


def _quality_score(flat: dict, has_judge: bool) -> float:
    weights = WEIGHTS_WITH_JUDGE if has_judge else WEIGHTS_NO_JUDGE
    score = 0.0
    used = 0.0
    for k, w in weights.items():
        v = flat.get(k)
        if v is None or v == 0:
            continue
        score += w * v
        used += w
    return score / used if used else 0.0


@dataclass
class _Check:
    name: str
    value: float
    threshold: float | None
    higher_is_better: bool
    pass_: bool
    layer: str

    @property
    def gap(self) -> float:
        if self.threshold is None:
            return 0.0
        return (self.threshold - self.value) if self.higher_is_better else (self.value - self.threshold)


def _gate_checks(flat: dict, th: Threshold, has_judge: bool) -> list[_Check]:
    checks = [
        _Check("Routing accuracy",     flat["accuracy"],          th.accuracy,          True,  flat["accuracy"]          >= th.accuracy,          "routing"),
        _Check("Routing macro-F1",     flat["macro_f1"],          th.macro_f1,          True,  flat["macro_f1"]          >= th.macro_f1,          "routing"),
        _Check("Refusal precision",    flat["refusal_precision"], th.refusal_precision, True,  flat["refusal_precision"] >= th.refusal_precision, "refusal"),
        _Check("Refusal recall",       flat["refusal_recall"],    th.refusal_recall,    True,  flat["refusal_recall"]    >= th.refusal_recall,    "refusal"),
        _Check("Citation recall",      flat["citation_recall"],   th.citation_recall,   True,  flat["citation_recall"]   >= th.citation_recall,   "retrieval"),
        _Check("Token-F1",             flat["token_f1"],          th.token_f1,          True,  flat["token_f1"]          >= th.token_f1,          "similarity"),
        _Check("ROUGE-L F1",           flat["rouge_l_f1"],        th.rouge_l_f1,        True,  flat["rouge_l_f1"]        >= th.rouge_l_f1,        "similarity"),
        _Check("p95 latency (s)",      flat["latency_p95"],       th.p95_latency_s,     False, flat["latency_p95"]       <= th.p95_latency_s,     "system"),
    ]
    if has_judge:
        checks += [
            _Check("Faithfulness",       flat["faithfulness"],       th.faithfulness,       True, flat["faithfulness"]       >= th.faithfulness,       "judge"),
            _Check("Answer relevancy",   flat["answer_relevancy"],   th.answer_relevancy,   True, flat["answer_relevancy"]   >= th.answer_relevancy,   "judge"),
            _Check("Context precision",  flat["context_precision"],  th.context_precision,  True, flat["context_precision"]  >= th.context_precision,  "judge"),
            _Check("Context recall",     flat["context_recall"],     th.context_recall,     True, flat["context_recall"]     >= th.context_recall,     "judge"),
            # answer_correctness has no separate threshold — we synthesise one
            _Check("Answer correctness", flat["answer_correctness"], 0.65, True,
                   flat["answer_correctness"] >= 0.65, "judge"),
        ]
    return checks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def consolidate(run_dir: Path) -> dict[str, Any]:
    rep_path = run_dir / "report.json"
    if not rep_path.exists():
        raise FileNotFoundError(f"No report.json in {run_dir}")
    report = json.loads(rep_path.read_text(encoding="utf-8"))

    # Extract model name from system report if available
    model_name = "Unknown SLM"
    if report:
        first_tier = next(iter(report.values()))
        model_name = _path(first_tier, "system", "model_name") or _path(first_tier, "system", "model") or "Unknown SLM"

    out: dict[str, Any] = {"run_dir": str(run_dir), "tiers": {}, "model_name": model_name}

    md = ["# Consolidated Evaluation Report", ""]
    md.append(f"**Model**: `{model_name}`")
    md.append(f"**Run**: `{run_dir.name}`")
    md.append("")

    for tier, rep in report.items():
        flat = _flatten(rep)
        has_judge = bool(rep.get("ragas")) and flat["answer_correctness"] > 0
        th = THRESHOLDS[tier]

        checks = _gate_checks(flat, th, has_judge)
        n_pass = sum(1 for c in checks if c.pass_)
        n_total = len(checks)
        qscore = _quality_score(flat, has_judge)

        out["tiers"][tier] = {
            "n_pairs": flat["n"],
            "judge_used": has_judge,
            "quality_score": qscore,
            "gate_pass": n_pass,
            "gate_total": n_total,
            "metrics": flat,
            "failures": [
                {"name": c.name, "value": c.value, "threshold": c.threshold,
                 "gap": c.gap, "layer": c.layer}
                for c in checks if not c.pass_
            ],
        }

        # ---- markdown digest -------------------------------------------------
        md.append(f"## Tier: `{tier}` ({flat['n']} pairs)")
        md.append("")
        md.append(f"- **Judge LLM used**: {'✅ yes (answer_correctness available)' if has_judge else '❌ no (deterministic only)'}")
        md.append(f"- **Quality Score** (weighted): **{qscore:.3f}** / 1.000")
        md.append(f"- **Gate**: {n_pass}/{n_total} thresholds met → "
                  f"{'**PASS**' if n_pass == n_total else '**FAIL**'}")
        md.append(f"- **Errors**: {flat['errors']}/{flat['n']}")
        md.append(f"- **Cost**: ${flat['cost_usd']:.4f}    "
                  f"**Latency p50/p95**: {flat['latency_p50']:.1f}s / {flat['latency_p95']:.1f}s")
        md.append("")

        md.append("### Layer-by-layer")
        md.append("")
        md.append("| Layer | Metric | Value | Threshold | Status |")
        md.append("|---|---|---:|---:|:---:|")
        for c in checks:
            arrow = "↑" if c.higher_is_better else "↓"
            status = "✅ PASS" if c.pass_ else "❌ FAIL"
            tval = f"{c.threshold:.2f}" if c.threshold is not None else "—"
            md.append(f"| {c.layer} | {c.name} {arrow} | {c.value:.3f} | {tval} | {status} |")
        md.append("")

        # raw metrics that don't have thresholds
        md.append("### Other metrics (no gate)")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---:|")
        for k in ["macro_precision", "macro_recall", "weighted_f1",
                  "precision_at_k", "recall_at_k", "ndcg_at_k",
                  "exact_match", "sbert_cosine", "bertscore_f1"]:
            v = flat.get(k, 0.0)
            md.append(f"| {k} | {v:.3f} |")
        md.append("")

        if has_judge:
            md.append("### Judge metrics (Ragas)")
            md.append("")
            md.append("| Metric | Value | Interpretation |")
            md.append("|---|---:|---|")
            md.append(f"| answer_correctness | {flat['answer_correctness']:.3f} | RN ≈ GT — claim-level F1 (the headline) |")
            md.append(f"| faithfulness       | {flat['faithfulness']:.3f} | answer grounded in retrieved context |")
            md.append(f"| answer_relevancy   | {flat['answer_relevancy']:.3f} | answer addresses the question |")
            md.append(f"| context_precision  | {flat['context_precision']:.3f} | retrieved chunks are useful |")
            md.append(f"| context_recall     | {flat['context_recall']:.3f} | retrieval covers the ground truth |")
            md.append("")

        # top failures by gap
        fails = [c for c in checks if not c.pass_]
        fails.sort(key=lambda c: c.gap, reverse=True)
        if fails:
            md.append("### Top failures by gap")
            md.append("")
            md.append("| # | Layer | Metric | Value | Threshold | Gap |")
            md.append("|---:|---|---|---:|---:|---:|")
            for i, c in enumerate(fails[:5], 1):
                md.append(f"| {i} | {c.layer} | {c.name} | {c.value:.3f} | "
                          f"{c.threshold:.2f} | {c.gap:+.3f} |")
            md.append("")

    out["markdown"] = "\n".join(md)
    return out


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="Path to a run directory containing report.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir for consolidated.{md,json} (default: same as run_dir)")
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    result = consolidate(args.run_dir)
    (out_dir / "consolidated.md").write_text(result["markdown"], encoding="utf-8")
    (out_dir / "consolidated.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "markdown"},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(result["markdown"])


if __name__ == "__main__":
    main()
