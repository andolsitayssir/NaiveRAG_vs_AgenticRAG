"""Judge-only re-scorer.

Given an existing run directory containing `traces.json`, run *only* the
Ragas judge phase against the saved RN/GT/contexts — no need to re-execute
the RAG pipeline. Useful when:
  - the judge crashed / was cancelled
  - you want to swap the judge model (e.g. OpenRouter Sonnet → OpenAI direct)
  - you want to add a new Ragas metric to an old run

Usage:
    OPENAI_API_KEY=sk-... \\
    python -m eval.judge_only reports/naive_ref_judge_mini \\
           --judge-provider openai-direct \\
           --judge-model openai/gpt-4o-mini

The script:
  1. Reads `traces.json` and rebuilds Trace objects
  2. Loads pairs from `test/<tier>/` to get questions + ground_truths
  3. Calls `eval.metrics.ragas_bridge.ragas_report` with the chosen judge
  4. Merges the new ragas block into the existing `report.json`
  5. Re-emits `consolidated.md` and `consolidated.json`

Direct-OpenAI path is much faster than OpenRouter for short judge calls
(no proxy hop) — typically 1-2 s/iter instead of 7-10 s/iter.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

from eval.parser import load_tier
from eval.runner import Trace


def _load_traces(run_dir: Path) -> list[Trace]:
    p = run_dir / "traces.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for d in raw:
        out.append(Trace(
            pair_id=d["pair_id"], tier=d["tier"], bucket=d["bucket"],
            tools_called=d.get("tools_called", []),
            tool_kwargs=d.get("tool_kwargs", []),
            predicted_ua=d.get("predicted_ua"),
            is_refusal=d.get("is_refusal", False),
            retrieved_sources=d.get("retrieved_sources", []),
            retrieved_chunks=d.get("retrieved_chunks", []),
            answer=d.get("answer", ""),
            ground_truth=d.get("ground_truth", ""),
            question=d.get("question", ""),
            latency_s=d.get("latency_s", 0.0),
            ttft_s=d.get("ttft_s"),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
            model_name=d.get("model_name", ""),
            embed_model=d.get("embed_model", ""),
            iterations=d.get("iterations", 1),
            error=d.get("error"),
            ragas=d.get("ragas"),
            ragas_error=d.get("ragas_error"),
            similarity=d.get("similarity"),
            similarity_error=d.get("similarity_error"),
        ))
    return out


def _build_judge(provider: str, model: str, temp: float):
    """Build a Ragas-compatible LLM. 'openai-direct' bypasses OpenRouter
    and hits api.openai.com directly — significantly faster for short calls."""
    from ragas.llms import LangchainLLMWrapper

    if provider == "openai-direct":
        from langchain_openai import ChatOpenAI
        # Strip any "openai/" prefix that came from OpenRouter conventions.
        if model.startswith("openai/"):
            model = model[len("openai/"):]
        llm = ChatOpenAI(
            model=model, temperature=temp,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60, max_retries=2,
        )
        return LangchainLLMWrapper(llm)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model, temperature=temp,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=60, max_retries=2,
        )
        return LangchainLLMWrapper(llm)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        return LangchainLLMWrapper(llm)

    raise ValueError(f"Unknown judge provider: {provider}")


def _build_embeddings():
    """Some Ragas metrics need embeddings. Use local sentence-transformers."""
    from eval.metrics.judge_embeddings import build_judge_embeddings
    return build_judge_embeddings()


def _build_dataset(pairs, traces_by_id):
    from datasets import Dataset
    rows = []
    skipped = 0
    for p in pairs:
        t = traces_by_id.get(p.id)
        if t is None or t.error or not p.ground_truth or not t.answer:
            skipped += 1
            continue
        rows.append({
            "pair_id":     p.id,
            "question":     p.question,
            "answer":       t.answer,
            "contexts":     t.retrieved_chunks or [""],
            "ground_truth": p.ground_truth,
        })
    print(f"[judge_only] dataset: {len(rows)} usable, {skipped} skipped")
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--judge-provider", default="openrouter",
                    choices=["openai-direct", "openrouter", "google"])
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="RAGAS parallel workers (overrides RAGAS_MAX_WORKERS env var)")
    ap.add_argument("--metrics", default="all",
                    help="Comma-separated subset: faithfulness, answer_relevancy, "
                         "context_precision, context_recall, answer_correctness, all")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        sys.exit(f"no such dir: {run_dir}")

    print(f"[judge_only] run_dir = {run_dir}")
    print(f"[judge_only] judge   = {args.judge_provider} :: {args.judge_model} (T={args.temp})")

    traces = _load_traces(run_dir)
    print(f"[judge_only] loaded {len(traces)} traces")

    # Group traces by tier and load matching pairs
    by_tier: dict[str, list[Trace]] = {}
    for t in traces:
        by_tier.setdefault(t.tier, []).append(t)

    judge = _build_judge(args.judge_provider, args.judge_model, args.temp)
    emb = _build_embeddings()

    from ragas import evaluate
    # Note: Ragas metrics are objects in 0.4+.
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    # Filter metrics per --metrics argument
    def _relevancy():
        try:
            return AnswerRelevancy(llm=judge, embeddings=emb, strictness=1)
        except TypeError:
            return AnswerRelevancy(llm=judge, embeddings=emb)

    def _correctness():
        try:
            return AnswerCorrectness(llm=judge, embeddings=emb, strictness=1)
        except TypeError:
            return AnswerCorrectness(llm=judge, embeddings=emb)

    _all = {
        "faithfulness":       Faithfulness(llm=judge),
        "answer_relevancy":   _relevancy(),
        "context_precision":  ContextPrecision(llm=judge),
        "context_recall":     ContextRecall(llm=judge),
        "answer_correctness": _correctness(),
    }
    if args.metrics == "all":
        active_metrics = list(_all.values())
        active_names = list(_all.keys())
    else:
        wanted = [m.strip() for m in args.metrics.split(",")]
        active_metrics = [_all[m] for m in wanted if m in _all]
        active_names = [m for m in wanted if m in _all]
    print(f"[judge_only] active metrics = {active_names}")

    # Load (or initialize) report.json — we'll merge ragas back in.
    rep_path = run_dir / "report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}

    # If report.json is missing or incomplete, recompute the deterministic
    # metrics from the saved traces. This is the "killed mid-flight" recovery path.
    from dataclasses import asdict
    from eval.metrics.classification import routing_report
    from eval.metrics.refusal import refusal_report
    from eval.metrics.retrieval import retrieval_report
    from eval.metrics.system import system_report
    from eval.metrics.similarity import similarity_report

    for tier, tier_traces in by_tier.items():
        print(f"\n[judge_only] === tier={tier}  pairs={len(tier_traces)} ===")
        pairs = load_tier(tier)
        pairs_by_id = {p.id: p for p in pairs}
        ordered_pairs = [pairs_by_id[t.pair_id] for t in tier_traces if t.pair_id in pairs_by_id]
        traces_by_id = {t.pair_id: t for t in tier_traces}

        # 1) deterministic layers (recompute if missing)
        if tier not in report or "routing" not in report.get(tier, {}):
            print(f"[judge_only] recomputing deterministic metrics for tier={tier}")
            routing = routing_report(ordered_pairs, tier_traces)
            refusal = refusal_report(ordered_pairs, tier_traces)
            retrieval = retrieval_report(ordered_pairs, tier_traces)
            system = system_report(tier_traces)
            similarity = similarity_report(
                ordered_pairs, tier_traces,
                use_bertscore=False, use_sbert=False,    # match original run
            )
            report[tier] = {
                "tier": tier,
                "n_pairs": len(tier_traces),
                "routing":   asdict(routing),
                "refusal":   asdict(refusal),
                "retrieval": asdict(retrieval),
                "system":    asdict(system),
                "similarity": asdict(similarity),
                "ragas": None,
            }

        # 2) judge layer (Ragas)
        ds = _build_dataset(ordered_pairs, traces_by_id)
        if len(ds) == 0:
            print(f"[judge_only] tier={tier}: nothing to judge, skipping ragas")
            continue

        # Force n=1 on metrics if applicable
        for m in active_metrics:
            if hasattr(m, "n"):
                m.n = 1

        t0 = time.perf_counter()
        from ragas.run_config import RunConfig
        max_workers = args.workers if args.workers is not None else int(os.getenv("RAGAS_MAX_WORKERS", "1"))
        timeout = int(os.getenv("RAGAS_TIMEOUT_S", "120"))
        results = evaluate(
            ds, metrics=active_metrics,
            llm=judge, embeddings=emb,
            raise_exceptions=False,
            run_config=RunConfig(max_workers=max_workers, timeout=timeout)
        )
        dt = time.perf_counter() - t0
        df = results.to_pandas()
        print(f"[judge_only] tier={tier} judged in {dt:.0f}s ({len(ds)} pairs)")

        # Update per-pair ragas in traces using pair_id mapping.
        if "pair_id" in df.columns:
            for _, row in df.iterrows():
                pid = row.get("pair_id")
                if not pid:
                    continue
                t = traces_by_id.get(pid)
                if t is None:
                    continue
                t.ragas = t.ragas or {}
                for name in active_names:
                    if name in row:
                        try:
                            val = float(row[name])
                            t.ragas[name] = None if val != val else val  # NaN check
                        except Exception:
                            t.ragas[name] = None

        def avg(col):
            s = df[col].dropna() if col in df.columns else None
            return float(s.mean()) if s is not None and len(s) else 0.0

        # Preserve previously-computed metrics if any (additive judge runs)
        ragas_block = (report.get(tier, {}).get("ragas") or {}).copy() if isinstance(
            report.get(tier, {}).get("ragas"), dict) else {}
        for name in active_names:
            ragas_block[name] = avg(name)
        ragas_block["n_samples"] = len(df)
        print(f"[judge_only] ragas[{tier}] = {json.dumps(ragas_block, indent=2)}")

        report[tier]["ragas"] = ragas_block

        # also persist the per-pair judge scores for drill-down
        df.to_csv(run_dir / f"ragas_per_pair_{tier}.csv", index=False)

    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[judge_only] report.json updated -> {rep_path}")

    # Persist updated traces with per-pair ragas.
    try:
        from eval.reporting.writers import write_traces_json
        write_traces_json(traces, run_dir / "traces.json")
        print(f"[judge_only] traces.json updated with per-pair ragas")
    except Exception as e:  # noqa: BLE001
        print(f"[judge_only] traces.json update skipped: {e}")

    # Regenerate consolidated.md / consolidated.json
    try:
        from eval.reporting.consolidated import consolidate
        cons = consolidate(run_dir)
        (run_dir / "consolidated.md").write_text(cons["markdown"], encoding="utf-8")
        (run_dir / "consolidated.json").write_text(
            json.dumps({k: v for k, v in cons.items() if k != "markdown"},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[judge_only] consolidated.md + consolidated.json written")
    except Exception as e:  # noqa: BLE001
        print(f"[judge_only] consolidated generation skipped: {e}")


if __name__ == "__main__":
    main()
