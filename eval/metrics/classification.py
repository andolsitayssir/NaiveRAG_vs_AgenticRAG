"""Classification metrics for UA routing: Accuracy, Precision, Recall, F1.

    Treats each pair as a single-label classification over ALL_LABELS = UA-1..UA-9 + NONE.
Multi-accept cases (e.g. expected_ua = 'UA-1|UA-9') are handled by matching
the predicted class against the set of acceptable classes — if the prediction
is in the set, it counts as correct for *any* of the acceptable classes,
and we attribute it to the first one for per-class breakdowns (deterministic).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from eval.config import ALL_LABELS, OOS_LABEL
from eval.runner import Trace
from eval.parser import Pair


@dataclass
class RoutingReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]   # class -> {precision, recall, f1, support}
    confusion: list[list[int]]               # ALL_LABELS × ALL_LABELS
    labels: list[str] = field(default_factory=lambda: list(ALL_LABELS))


def _resolve_truth(expected_uas: list[str], predicted: str | None) -> str:
    """If prediction is in the accepted set, we treat it as the truth label
    so it doesn't get counted against us. Otherwise, we pin the truth to the
    first expected UA (deterministic tie-breaker)."""
    if not expected_uas:
        return OOS_LABEL
    if predicted and predicted in expected_uas:
        return predicted
    return expected_uas[0]


def _map_prediction(predicted: str | None) -> str:
    """Any non-UA / non-NONE prediction is collapsed to OOS_LABEL for the
    confusion matrix to keep the label set closed. 'GLOBAL' -> OOS_LABEL."""
    if predicted in ALL_LABELS:
        return predicted
    return OOS_LABEL


def routing_report(pairs: Iterable[Pair], traces: Iterable[Trace]) -> RoutingReport:
    pairs = list(pairs)
    traces = {t.pair_id: t for t in traces}

    y_true: list[str] = []
    y_pred: list[str] = []
    for p in pairs:
        t = traces.get(p.id)
        pred = _map_prediction(t.predicted_ua if t else None)
        y_pred.append(pred)
        y_true.append(_resolve_truth(p.expected_uas, pred))

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, average="weighted", zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0,
    )
    per_class = {
        lbl: {
            "precision": report_dict[lbl]["precision"],
            "recall":    report_dict[lbl]["recall"],
            "f1":        report_dict[lbl]["f1-score"],
            "support":   report_dict[lbl]["support"],
        }
        for lbl in ALL_LABELS
    }
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS).tolist()

    return RoutingReport(
        accuracy=acc,
        macro_precision=p_macro,
        macro_recall=r_macro,
        macro_f1=f1_macro,
        weighted_f1=f1_weighted,
        per_class=per_class,
        confusion=cm,
    )
