"""Parse test/<tier>/UA-*.md and out_of_scope.md into Pair objects."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from eval.config import TEST_DIR, TIERS, UA_IDS, OOS_LABEL


@dataclass
class Pair:
    id: str
    tier: str                # "reference" | "medium" | "advanced"
    bucket: str              # "UA-1" | ... | "UA-9" | "OOS"
    question: str
    expected_ua_raw: str     # as written in the file, e.g. "UA-1|UA-9" or "NONE"
    expected_uas: list[str]  # parsed — list of acceptable UAs, or ["NONE"]
    must_cite: list[str]
    tags: list[str]
    ground_truth: str


# ---------------------------------------------------------------------------
# Markdown block regex
# ---------------------------------------------------------------------------

_H2 = re.compile(r"^## ([A-Z0-9\-]+)\s*$", re.MULTILINE)
_FIELD = re.compile(
    r"^-\s*\*\*(?P<key>[a-zA-Z_]+):\*\*\s*(?P<val>.*?)$",
    re.MULTILINE,
)
_GT = re.compile(
    r"^-\s*\*\*ground_truth:\*\*\s*\n(?P<body>(?:[ \t]*>\s?.*\n?)+)",
    re.MULTILINE,
)


def _split_list(raw: str) -> list[str]:
    raw = raw.strip().strip("`")
    if not raw or raw == "—":
        return []
    parts = re.split(r"`\s*,\s*`|\s*,\s*", raw)
    return [p.strip("`" ).strip() for p in parts if p.strip()]


def _parse_expected_ua(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw or raw.upper().startswith("NONE"):
        return [OOS_LABEL]
    # forms: "UA-1", "UA-1|UA-9", "GLOBAL", "NONE_OR_UA-5"
    if raw.upper() == "GLOBAL":
        return ["GLOBAL"]
    if raw.upper().startswith("NONE_OR_"):
        # refusal acceptable, OR the listed UA
        rest = raw.split("_OR_", 1)[1]
        return [OOS_LABEL] + [r.strip() for r in rest.split("|")]
    return [r.strip() for r in raw.split("|") if r.strip()]


def parse_file(path: Path, tier: str, bucket: str) -> list[Pair]:
    text = path.read_text(encoding="utf-8")
    pairs: list[Pair] = []
    matches = list(_H2.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        fields = {mm.group("key"): mm.group("val") for mm in _FIELD.finditer(block)}
        if "question" not in fields:
            continue

        gt_match = _GT.search(block)
        gt = ""
        if gt_match:
            # strip optional leading spaces + '>' + optional space
            gt = re.sub(r"^[ \t]*>\s?", "", gt_match.group("body"), flags=re.MULTILINE).strip()

        raw_exp = fields.get("expected_ua", "")
        expected_uas = _parse_expected_ua(raw_exp)

        pairs.append(Pair(
            id=m.group(1),
            tier=tier,
            bucket=bucket,
            question=fields.get("question", "").strip(),
            expected_ua_raw=raw_exp,
            expected_uas=expected_uas,
            must_cite=_split_list(fields.get("must_cite", "")),
            tags=[t.strip() for t in fields.get("tags", "").split(",") if t.strip()],
            ground_truth=gt,
        ))
    return pairs


def load_tier(tier: str) -> list[Pair]:
    """Load all pairs for one tier. Order: UA-1..UA-9, then out_of_scope."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}")

    tier_dir = TEST_DIR / tier
    pairs: list[Pair] = []
    for ua in UA_IDS:
        f = tier_dir / f"{ua}.md"
        if f.exists():
            pairs.extend(parse_file(f, tier, ua))
    oos = tier_dir / "out_of_scope.md"
    if oos.exists():
        pairs.extend(parse_file(oos, tier, "OOS"))
    return pairs


def load_all() -> dict[str, list[Pair]]:
    return {t: load_tier(t) for t in TIERS}
