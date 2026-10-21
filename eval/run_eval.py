"""Entry point: run the eval program.

Usage:
    python -m eval.run_eval --tier reference
    python -m eval.run_eval --tier all --workers 8 --out-dir reports/run_01
    python -m eval.run_eval --tier reference --limit 5 --mock
    python -m eval.run_eval --tier all --ragas --out-dir reports/nightly

Exit codes:
    0  all thresholds met
    1  at least one threshold failed (release gate)
    2  fatal error
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from datetime import datetime
from pathlib import Path

from eval.config import TIERS, THRESHOLDS
from eval.parser import load_tier
from eval.runner import run_pairs, MockSvc
from eval.metrics import compute_tier, TierReport
from eval.reporting.console import print_tier
from eval.reporting.writers import write_json, write_csv, write_html, write_traces_json


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def gate_fail(r: TierReport) -> list[str]:
    th = THRESHOLDS[r.tier]
    fails = []
    def check(name, v, limit, higher=True):
        ok = (v >= limit) if higher else (v <= limit)
        if not ok:
            fails.append(f"[{r.tier}] {name}={v:.3f} vs threshold {limit:.2f}")

    check("accuracy", r.routing.accuracy, th.accuracy)
    check("macro_f1", r.routing.macro_f1, th.macro_f1)
    check("refusal_precision", r.refusal.precision, th.refusal_precision)
    check("refusal_recall", r.refusal.recall, th.refusal_recall)
    check("citation_recall", r.retrieval.citation_recall, th.citation_recall)
    check("p95_latency", r.system.latency_p95_s, th.p95_latency_s, higher=False)

    if r.similarity:
        check("token_f1", r.similarity["token_f1"], th.token_f1)
        check("rouge_l_f1", r.similarity["rouge_l_f1"], th.rouge_l_f1)
        if r.similarity.get("sbert_cosine"):
            check("sbert_cosine", r.similarity["sbert_cosine"], th.sbert_cosine)
        if r.similarity.get("bertscore_f1") is not None:
            check("bertscore_f1", r.similarity["bertscore_f1"], th.bertscore_f1)

    if r.ragas:
        check("faithfulness", r.ragas["faithfulness"], th.faithfulness)
        check("answer_relevancy", r.ragas["answer_relevancy"], th.answer_relevancy)
        check("context_precision", r.ragas["context_precision"], th.context_precision)
        check("context_recall", r.ragas["context_recall"], th.context_recall)

    if r.system.over_cap_iterations > 0:
        fails.append(f"[{r.tier}] iterations>4: {r.system.over_cap_iterations}")

    return fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(args):
    tiers = list(TIERS) if args.tier == "all" else [args.tier]

    # Build the Agentic RAG service
    if args.mock:
        # build mock service from the pairs we're about to run
        pairs_for_mock = {}
        for t in tiers:
            for p in load_tier(t):
                pairs_for_mock[p.question] = p
        svc = MockSvc({p.id: p for p in pairs_for_mock.values()})
    else:
        # real service — pluggable: naive baseline or agentic
        from eval.targets import build_target
        svc = build_target(args.target)

    reports: dict[str, TierReport] = {}
    all_traces = []

    for tier in tiers:
        pairs = load_tier(tier)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"\n>>> Running tier={tier}  n={len(pairs)}  ragas={args.ragas}  mock={args.mock}", flush=True)
        t0 = time.perf_counter()

        # Instrument run_pairs with a progress callback so the UI sees
        # a heartbeat every ~5 pairs (otherwise ingestion + LLM calls are silent).
        done = {"n": 0, "err": 0}
        total = len(pairs)

        from eval.runner import run_pair  # type: ignore

        sem = asyncio.Semaphore(args.workers)

        def _fmt(s: float) -> str:
            s = max(0, int(s))
            if s >= 3600:
                return f"{s//3600}h{(s%3600)//60:02d}m{s%60:02d}s"
            return f"{s//60}m{s%60:02d}s"

        async def _run(p):
            async with sem:
                t = await run_pair(svc, p)
                done["n"] += 1
                if getattr(t, "error", None):
                    done["err"] += 1
                elapsed = time.perf_counter() - t0
                rate = done["n"] / max(elapsed, 1e-6)
                eta = (total - done["n"]) / max(rate, 1e-6)
                pct = 100.0 * done["n"] / total
                print(f"    [{tier}] {done['n']}/{total} ({pct:5.1f}%)  "
                      f"elapsed={_fmt(elapsed)}  remaining≈{_fmt(eta)}  "
                      f"err={done['err']}", flush=True)
                return t

        _traces: list = []
        _traces = await asyncio.gather(*[_run(p) for p in pairs])
        traces = _traces
        print(f"    wall={time.perf_counter()-t0:.1f}s  "
              f"errors={sum(1 for t in traces if t.error)}", flush=True)

        # Persist traces immediately after the RAG phase so they survive a
        # judge-phase crash / cancellation. The final write later overwrites this.
        out_dir_early = Path(args.out_dir) if args.out_dir else None
        if out_dir_early:
            out_dir_early.mkdir(parents=True, exist_ok=True)
            try:
                from eval.reporting.writers import write_traces_json as _wtj
                _wtj(traces, out_dir_early / "traces.json")
                print(f"    [{tier}] traces persisted ({len(traces)} pairs) -> {out_dir_early/'traces.json'}")
            except Exception as _e:
                print(f"    [warn] early trace dump skipped: {_e}")

        rep = compute_tier(
            tier, pairs, traces,
            with_similarity=not args.no_similarity,
            with_ragas=args.ragas,
            use_bertscore=not args.no_bertscore,
            use_sbert=not args.no_sbert,
        )
        reports[tier] = rep
        all_traces.extend(traces)
        print_tier(rep)

    # Write outputs
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports, out_dir / "report.json")
    write_csv(all_traces, out_dir / "traces.csv")
    write_traces_json(all_traces, out_dir / "traces.json")
    write_html(reports, out_dir / "report.html")
    # consolidated digest (deterministic + judge in one place)
    try:
        from eval.reporting.consolidated import consolidate
        cons = consolidate(out_dir)
        (out_dir / "consolidated.md").write_text(cons["markdown"], encoding="utf-8")
        import json as _json
        (out_dir / "consolidated.json").write_text(
            _json.dumps({k: v for k, v in cons.items() if k != "markdown"},
                        indent=2, ensure_ascii=False), encoding="utf-8",
        )
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] consolidated report skipped: {_e}")
    print(f"\n>>> Reports written to {out_dir}")

    # Gate check
    all_fails = []
    for rep in reports.values():
        all_fails.extend(gate_fail(rep))
    if all_fails:
        print("\n>>> GATE: FAIL")
        for f in all_fails:
            print(f"    - {f}")
        return 1
    print("\n>>> GATE: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS) + ["all"], default="reference")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ragas", action="store_true",
                    help="Enable judge-based RAG metrics (faithfulness, relevancy, "
                         "ctx-P/R). Slower + LLM cost. Pin judge in eval/config.py.")
    ap.add_argument("--no-similarity", action="store_true",
                    help="Skip judge-free answer-similarity metrics (EM/F1/ROUGE/BERTScore).")
    ap.add_argument("--no-bertscore", action="store_true",
                    help="Skip BERTScore (downloads xlm-roberta-base on first use).")
    ap.add_argument("--no-sbert", action="store_true",
                    help="Skip SBERT cosine (downloads MiniLM-L12-v2 on first use).")
    ap.add_argument("--mock", action="store_true", help="Use a deterministic mock RAG service")
    ap.add_argument("--target", choices=["naive", "agentic"], default="naive",
                    help="Which RAG to evaluate (ignored if --mock).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    try:
        code = asyncio.run(_run(args))
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
