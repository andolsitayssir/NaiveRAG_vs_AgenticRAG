"""Evaluation configuration — paths, thresholds, judge, cost table."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# Load env vars early
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"
TIERS = ("reference", "medium", "advanced")

UA_IDS = [f"UA-{i}" for i in range(1, 10)]
OOS_LABEL = "NONE"   # reserved class for out-of-scope pairs
ALL_LABELS = UA_IDS + [OOS_LABEL]


# ---------------------------------------------------------------------------
# Release-gate thresholds per tier (mirror of the design doc §8.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Threshold:
    # classification (judge-free, mandatory)
    accuracy: float
    macro_f1: float
    refusal_precision: float
    refusal_recall: float
    citation_recall: float
    # answer similarity (judge-free, recommended)
    token_f1: float
    rouge_l_f1: float
    sbert_cosine: float
    bertscore_f1: float
    # judge-based (optional, nightly)
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    # system
    p95_latency_s: float


# Thresholds = per-tier release gates. A run passes if every populated metric
# meets its threshold. Metrics that weren't computed (e.g. ragas without --ragas)
# are skipped in the gate check.
THRESHOLDS: dict[str, Threshold] = {
    # tier        acc  mF1  rP   rR   cR   tF1  rL   sbC  bsF  F    aR   cP   cR   p95
    "reference": Threshold(0.95, 0.93, 0.95, 0.90, 0.90,
                           0.55, 0.45, 0.80, 0.85,
                           0.90, 0.85, 0.80, 0.80, 6.0),
    "medium":    Threshold(0.90, 0.88, 0.95, 0.90, 0.85,
                           0.45, 0.35, 0.75, 0.80,
                           0.85, 0.80, 0.70, 0.75, 8.0),
    "advanced":  Threshold(0.75, 0.72, 0.95, 0.80, 0.75,
                           0.35, 0.25, 0.70, 0.75,
                           0.80, 0.75, 0.60, 0.65, 10.0),
}


# ---------------------------------------------------------------------------
# Regression gate: a new run is rejected if ANY metric drops by MORE than this
# many percentage points in ANY tier vs. the baseline.
# ---------------------------------------------------------------------------

MAX_REGRESS_POINTS = 2.0


# ---------------------------------------------------------------------------
# Ragas judge — pinned for reproducibility.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 2


def _judge_from_env() -> "JudgeConfig":
    """Allow env overrides without editing code.

    ACTIVIITY_JUDGE_PROVIDER = openrouter | ollama | google
    ACTIVIITY_JUDGE_MODEL    = openai/gpt-4o-mini | ...
    """
    import os
    return JudgeConfig(
        provider=os.getenv("ACTIVIITY_JUDGE_PROVIDER", "openrouter"),
        model=os.getenv("ACTIVIITY_JUDGE_MODEL", "openai/gpt-4o-mini"),
        temperature=float(os.getenv("ACTIVIITY_JUDGE_TEMPERATURE", "0.0")),
        max_retries=int(os.getenv("ACTIVIITY_JUDGE_MAX_RETRIES", "2")),
    )


JUDGE = _judge_from_env()

# Budget guard-rail: runner aborts a tier if estimated cost exceeds this.
# Set to 0 to disable.
import os as _os
BUDGET_USD = float(_os.getenv("ACTIVIITY_EVAL_BUDGET_USD", "0"))


# ---------------------------------------------------------------------------
# Cost table (USD / 1M tokens) — adjust as prices change.
# ---------------------------------------------------------------------------

COST_PER_MILLION: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "anthropic/claude-haiku-4-5":  {"in": 0.25, "out": 1.25},
    "openai/gpt-4o-mini":          {"in": 0.15, "out": 0.60},
    "mistralai/ministral-8b-2512": {"in": 0.10, "out": 0.10},
    "mistralai/mistral-7b-instruct-v0.1": {"in": 0.10, "out": 0.10},
    "z-ai/glm-4.7-flash":          {"in": 0.10, "out": 0.10},
    "openai/gpt-oss-20b":          {"in": 0.10, "out": 0.40},
    "openai/gpt-4o-mini":          {"in": 0.15, "out": 0.60},
   
    
    "qwen2.5:14b-instruct":        {"in": 0.00, "out": 0.00},   # on-prem
    "bge-m3":                      {"in": 0.00, "out": 0.00},
    "text-embedding-3-small":      {"in": 0.02, "out": 0.00},
}


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

RETRIEVAL_K = 5          # Precision@k / Recall@k / nDCG@k


# ---------------------------------------------------------------------------
# Refusal detection regex (must be aligned with defaults_fr.refusal_fr)
# ---------------------------------------------------------------------------

REFUSAL_PATTERNS = [
    r"(?i)\b(désolé|désolée)\b",
    r"(?i)\bje ne trouve pas\b",
    r"(?i)\bpas (cette|d['']) ?information\b",
    r"(?i)\babsente? du corpus\b",
    r"(?i)\bbase activiity\b.*\b(pas|n['']ai)\b",
    r"(?i)\bne dispose (pas|pas de)\b",
    r"(?i)\binformation.*?(absente?|manquante|disponible)\b",
]
