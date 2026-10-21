"""Refusal classification metrics: Precision, Recall, F1 on the refusal axis.

Binary task:
- y_true = 1 if the pair is out-of-scope (expected_uas contains 'NONE' and ONLY 'NONE')
- y_pred = 1 if the agent's answer matched the refusal pattern
For borderline pairs (e.g. expected_ua='NONE_OR_UA-5'), refusal is acceptable but
not required — such pairs are excluded from the strict binary calculation and
reported separately.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from eval.config import OOS_LABEL
from eval.parser import Pair
from eval.runner import Trace


def _is_strict_oos(p: Pair) -> bool:
    return p.expected_uas == [OOS_LABEL]

def _is_borderline(p: Pair) -> bool:
    return OOS_LABEL in p.expected_uas and len(p.expected_uas) > 1


@dataclass
class RefusalReport:
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    borderline_correct: int      # # borderline pairs where refusal OR valid UA answer was given
    borderline_total: int


def refusal_report(pairs: Iterable[Pair], traces: Iterable[Trace]) -> RefusalReport:
    pairs = list(pairs)
    traces = {t.pair_id: t for t in traces}

    y_true, y_pred = [], []
    tp = fp = tn = fn = 0
    borderline_total = borderline_correct = 0

    for p in pairs:
        t = traces.get(p.id)
        refused = t.is_refusal if t else False

        if _is_borderline(p):
            borderline_total += 1
            # Accept either: a refusal, or a response citing one of the
            # accepted non-NONE UAs. We rely on the routing trace for the
            # latter — already populated in Trace.predicted_ua.
            accepted = set(p.expected_uas) - {OOS_LABEL}
            ok = refused or (t and t.predicted_ua in accepted)
            borderline_correct += 1 if ok else 0
            continue

        is_oos = _is_strict_oos(p)
        y_true.append(1 if is_oos else 0)
        y_pred.append(1 if refused else 0)

        if is_oos and refused: tp += 1
        elif is_oos and not refused: fn += 1
        elif (not is_oos) and refused: fp += 1
        else: tn += 1

    if not y_true:
        return RefusalReport(0.0, 0.0, 0.0, 0.0, tp, fp, tn, fn,
                             borderline_correct, borderline_total)

    p_, r_, f1_, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return RefusalReport(
        precision=p_, recall=r_, f1=f1_, accuracy=acc,
        true_positives=tp, false_positives=fp,
        true_negatives=tn, false_negatives=fn,
        borderline_correct=borderline_correct,
        borderline_total=borderline_total,
    )
