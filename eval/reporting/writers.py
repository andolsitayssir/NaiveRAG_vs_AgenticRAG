"""JSON / CSV / HTML writers for evaluation results."""
from __future__ import annotations
import csv, json, html
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from eval.metrics import TierReport
from eval.runner import Trace


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def write_json(reports: dict[str, TierReport], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({tier: r.to_dict() for tier, r in reports.items()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_traces_json(traces: Iterable[Trace], out: Path):
    """Persist full per-pair traces (including retrieved_chunks) so Ragas can be
    added after-the-fact without re-executing the RAG pipeline."""
    out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict as _asdict
    out.write_text(
        json.dumps([_asdict(t) for t in traces], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CSV — one row per pair, for drill-down
# ---------------------------------------------------------------------------

def write_csv(traces: Iterable[Trace], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "tier", "bucket", "predicted_ua", "is_refusal",
        "latency_s", "ttft_s", "tokens_in", "tokens_out", "iterations",
        "error",
        "sim_exact_match", "sim_token_f1", "sim_token_precision", "sim_token_recall",
        "sim_rouge_l_f1", "sim_sbert_cosine", "sim_bertscore_f1", "similarity_error",
        "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
        "ragas_context_recall", "ragas_answer_correctness", "ragas_error",
        "question", "ground_truth", "answer",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for t in traces:
            row = asdict(t)

            sim = row.get("similarity")
            if isinstance(sim, dict):
                row["sim_exact_match"] = sim.get("exact_match")
                row["sim_token_f1"] = sim.get("token_f1")
                row["sim_token_precision"] = sim.get("token_precision")
                row["sim_token_recall"] = sim.get("token_recall")
                row["sim_rouge_l_f1"] = sim.get("rouge_l_f1")
                row["sim_sbert_cosine"] = sim.get("sbert_cosine")
                row["sim_bertscore_f1"] = sim.get("bertscore_f1")
            else:
                row["sim_exact_match"] = None
                row["sim_token_f1"] = None
                row["sim_token_precision"] = None
                row["sim_token_recall"] = None
                row["sim_rouge_l_f1"] = None
                row["sim_sbert_cosine"] = None
                row["sim_bertscore_f1"] = None

            ragas = row.get("ragas")
            if isinstance(ragas, dict):
                row["ragas_faithfulness"] = ragas.get("faithfulness")
                row["ragas_answer_relevancy"] = ragas.get("answer_relevancy")
                row["ragas_context_precision"] = ragas.get("context_precision")
                row["ragas_context_recall"] = ragas.get("context_recall")
                row["ragas_answer_correctness"] = ragas.get("answer_correctness")
            else:
                row["ragas_faithfulness"] = None
                row["ragas_answer_relevancy"] = None
                row["ragas_context_precision"] = None
                row["ragas_context_recall"] = None
                row["ragas_answer_correctness"] = None

            row["answer"] = (row.get("answer") or "").replace("\n", " ")[:500]
            row["ground_truth"] = (row.get("ground_truth") or "").replace("\n", " ")[:500]
            row["question"] = (row.get("question") or "").replace("\n", " ")[:300]
            w.writerow(row)


# ---------------------------------------------------------------------------
# HTML — self-contained dashboard
# ---------------------------------------------------------------------------

_HTML_HEAD = """<!doctype html><meta charset=utf-8>
<title>Activiity — RAG eval report</title>
<style>
body{font:14px -apple-system,sans-serif;max-width:1100px;margin:24px auto;padding:0 20px;color:#222}
h1{font-size:22px;margin-bottom:4px}
h2{font-size:17px;margin:30px 0 6px;padding-top:10px;border-top:1px solid #eee}
h3{font-size:14px;margin:16px 0 6px;color:#666}
table{border-collapse:collapse;width:100%;margin:8px 0}
td,th{padding:6px 10px;border:1px solid #e5e5e5;text-align:right;font-size:13px}
th{background:#f5f5f5;text-align:left}
td:first-child,th:first-child{text-align:left}
.pass{color:#0a7a0a;font-weight:600}
.fail{color:#b00;font-weight:600}
.tier{background:#f9f9fc;padding:12px;border-radius:8px;margin-bottom:18px}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px}
.small{color:#888;font-size:12px}
</style>
"""


def _pass_cell(v: float, th: float, higher=True) -> str:
    ok = (v >= th) if higher else (v <= th)
    cls = "pass" if ok else "fail"
    label = "PASS" if ok else "FAIL"
    return f'<td>{v:.3f}</td><td>{th:.2f}</td><td class="{cls}">{label}</td>'


def write_html(reports: dict[str, TierReport], out: Path):
    from eval.config import THRESHOLDS

    # Extract model name from system report if available
    model_name = "Unknown SLM"
    if reports:
        first_tier = next(iter(reports.values()))
        if hasattr(first_tier.system, "model_name") and first_tier.system.model_name:
            model_name = first_tier.system.model_name
        elif hasattr(first_tier.system, "model") and first_tier.system.model:
            model_name = first_tier.system.model

    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [_HTML_HEAD, f"<h1>Activiity — {html.escape(model_name)} report</h1>"]
    parts.append('<p class="small">Generated by <code>eval/run_eval.py</code>. '
                 'Thresholds from <code>eval/config.py</code>.</p>')

    for tier, r in reports.items():
        th = THRESHOLDS[tier]
        parts.append(f'<div class="tier"><h2>Tier: {html.escape(tier)} '
                     f'<span class="small">({r.n_pairs} pairs)</span></h2>')

        parts.append("<h3>Routing (UA classification)</h3><table>")
        parts.append("<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Result</th></tr>")
        parts.append("<tr><td>Accuracy</td>" + _pass_cell(r.routing.accuracy, th.accuracy) + "</tr>")
        parts.append("<tr><td>Macro-F1</td>" + _pass_cell(r.routing.macro_f1, th.macro_f1) + "</tr>")
        parts.append(f"<tr><td>Macro-Precision</td><td>{r.routing.macro_precision:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>Macro-Recall</td><td>{r.routing.macro_recall:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>Weighted-F1</td><td>{r.routing.weighted_f1:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append("</table>")

        parts.append("<h3>Per-class F1</h3><table>")
        parts.append("<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>")
        for cls, d in r.routing.per_class.items():
            parts.append(f"<tr><td>{cls}</td><td>{d['precision']:.3f}</td>"
                         f"<td>{d['recall']:.3f}</td><td>{d['f1']:.3f}</td>"
                         f"<td>{int(d['support'])}</td></tr>")
        parts.append("</table>")

        parts.append("<h3>Refusal</h3><table>")
        parts.append("<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Result</th></tr>")
        parts.append("<tr><td>Precision</td>" + _pass_cell(r.refusal.precision, th.refusal_precision) + "</tr>")
        parts.append("<tr><td>Recall</td>" + _pass_cell(r.refusal.recall, th.refusal_recall) + "</tr>")
        parts.append(f"<tr><td>F1</td><td>{r.refusal.f1:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>TP / FP / TN / FN</td><td colspan=3>"
                     f"{r.refusal.true_positives} / {r.refusal.false_positives} / "
                     f"{r.refusal.true_negatives} / {r.refusal.false_negatives}</td></tr>")
        parts.append("</table>")

        parts.append("<h3>Retrieval</h3><table>")
        parts.append(f"<tr><th>Metric (@{r.retrieval.k})</th><th>Value</th><th>Threshold</th><th>Result</th></tr>")
        parts.append(f"<tr><td>Precision@k</td><td>{r.retrieval.precision_at_k:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>Recall@k</td><td>{r.retrieval.recall_at_k:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>MRR</td><td>{r.retrieval.mrr:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append(f"<tr><td>nDCG@k</td><td>{r.retrieval.ndcg_at_k:.3f}</td><td>—</td><td>—</td></tr>")
        parts.append("<tr><td>Citation recall</td>" + _pass_cell(r.retrieval.citation_recall, th.citation_recall) + "</tr>")
        parts.append("</table>")

        if r.similarity:
            s = r.similarity
            parts.append("<h3>Answer similarity (judge-free)</h3><table>")
            parts.append("<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Result</th></tr>")
            parts.append(f"<tr><td>Exact Match</td><td>{s['exact_match']:.3f}</td><td>—</td><td>—</td></tr>")
            parts.append("<tr><td>Token-F1</td>" + _pass_cell(s["token_f1"], th.token_f1) + "</tr>")
            parts.append(f"<tr><td>Token P / R</td><td colspan=3>{s['token_precision']:.3f} / {s['token_recall']:.3f}</td></tr>")
            parts.append("<tr><td>ROUGE-L F1</td>" + _pass_cell(s["rouge_l_f1"], th.rouge_l_f1) + "</tr>")
            if s.get("sbert_cosine"):
                parts.append("<tr><td>SBERT cosine</td>" + _pass_cell(s["sbert_cosine"], th.sbert_cosine) + "</tr>")
            if s.get("bertscore_f1") is not None:
                parts.append("<tr><td>BERTScore F1</td>" + _pass_cell(s["bertscore_f1"], th.bertscore_f1) + "</tr>")
            parts.append("</table>")

        if r.ragas:
            parts.append("<h3>RAG quality (Ragas — judge-based)</h3><table>")
            parts.append("<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Result</th></tr>")
            parts.append("<tr><td>Faithfulness</td>" + _pass_cell(r.ragas["faithfulness"], th.faithfulness) + "</tr>")
            parts.append("<tr><td>Answer relevancy</td>" + _pass_cell(r.ragas["answer_relevancy"], th.answer_relevancy) + "</tr>")
            parts.append("<tr><td>Context precision</td>" + _pass_cell(r.ragas["context_precision"], th.context_precision) + "</tr>")
            parts.append("<tr><td>Context recall</td>" + _pass_cell(r.ragas["context_recall"], th.context_recall) + "</tr>")
            parts.append("</table>")

        parts.append("<h3>System</h3><table>")
        parts.append(f"<tr><td>Errors</td><td>{r.system.errors}/{r.system.n}</td><td>—</td><td>—</td></tr>")
        parts.append("<tr><td>Latency p95 (s)</td>" + _pass_cell(r.system.latency_p95_s, th.p95_latency_s, higher=False) + "</tr>")
        parts.append(f"<tr><td>Latency p50 / p99 (s)</td><td colspan=3>{r.system.latency_p50_s:.2f} / {r.system.latency_p99_s:.2f}</td></tr>")
        parts.append(f"<tr><td>Avg tokens in / out</td><td colspan=3>{r.system.avg_tokens_in:.0f} / {r.system.avg_tokens_out:.0f}</td></tr>")
        parts.append(f"<tr><td>Cost (USD)</td><td colspan=3>${r.system.cost_usd:.4f}</td></tr>")
        parts.append(f"<tr><td>Iterations > 4</td><td colspan=3>{r.system.over_cap_iterations}</td></tr>")
        parts.append("</table></div>")

    out.write_text("".join(parts), encoding="utf-8")
