"""System metrics: latency percentiles, tokens, cost, tool-call stats."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import median, mean
from typing import Iterable
import math

from eval.config import COST_PER_MILLION
from eval.runner import Trace


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class SystemReport:
    n: int
    errors: int
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    ttft_p50_s: float
    avg_tokens_in: float
    avg_tokens_out: float
    total_tokens_in: int
    total_tokens_out: int
    cost_usd: float
    avg_iterations: float
    over_cap_iterations: int      # # of traces with iterations > 4


def _cost(tokens_in: int, tokens_out: int, model: str) -> float:
    if not model or not isinstance(model, str):
        return 0.0

    # Clean the model name for better matching (lowercase and strip specific patterns)
    m = model.lower()
    
    # Exact match first
    if model in COST_PER_MILLION:
        p = COST_PER_MILLION[model]
        return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]
    
    # Fuzzy match (OpenRouter model names often have prefixes)
    for known_model, p in COST_PER_MILLION.items():
        if known_model.lower() in m or m in known_model.lower():
            return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]

    return 0.0


def system_report(traces: Iterable[Trace]) -> SystemReport:
    traces = list(traces)
    if not traces:
        return SystemReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0)

    lat = [t.latency_s for t in traces if not t.error]
    ttft = [t.ttft_s for t in traces if t.ttft_s is not None]
    tin = sum(t.tokens_in for t in traces)
    tout = sum(t.tokens_out for t in traces)

    cost = sum(_cost(t.tokens_in, t.tokens_out, t.model_name) for t in traces)

    return SystemReport(
        n=len(traces),
        errors=sum(1 for t in traces if t.error),
        latency_p50_s=_percentile(lat, 0.50),
        latency_p95_s=_percentile(lat, 0.95),
        latency_p99_s=_percentile(lat, 0.99),
        ttft_p50_s=_percentile(ttft, 0.50),
        avg_tokens_in=mean([t.tokens_in for t in traces]) if traces else 0.0,
        avg_tokens_out=mean([t.tokens_out for t in traces]) if traces else 0.0,
        total_tokens_in=tin,
        total_tokens_out=tout,
        cost_usd=cost,
        avg_iterations=mean([t.iterations for t in traces]) if traces else 0.0,
        over_cap_iterations=sum(1 for t in traces if t.iterations > 4),
    )
