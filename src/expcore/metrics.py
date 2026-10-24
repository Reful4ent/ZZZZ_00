"""Metrics with reject semantics (spec §6.1–6.2).

Q = pairs without not_applicable; TP/FP/TN/FN count accepted decisions only, rejects stay in N±
of their true class. A zero denominator gives null with a reason, never 0.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from expcore.errors import DataValidationError, MixedRelationError
from expcore.registry import Outcome

EMPTY_CLASS = "undefined_empty_class"
NO_ACCEPTED = "undefined_no_accepted"
POSITIVE_CLASSES = ("match", "defect")


@dataclass(frozen=True)
class MetricsRow:
    tp: int
    fp: int
    tn: int
    fn: int
    rej_pos: int
    rej_neg: int
    n_pos: int
    n_neg: int
    n_not_applicable: int
    coverage: float | None
    coverage_pos: float | None
    coverage_neg: float | None
    rho_rej: float | None
    recall_all: float | None
    fpr_all: float | None
    correct_all: float | None
    recall_cond: float | None
    fpr_cond: float | None
    correct_cond: float | None
    positive_class: str
    null_reasons: dict[str, str] = field(default_factory=dict)


def _labels(df: pd.DataFrame, positive_class: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(y_true, y_pred, accepted) over Q in the chosen positive-class encoding, plus n_not_applicable."""
    if positive_class not in POSITIVE_CLASSES:
        raise ValueError(f"positive_class must be one of {POSITIVE_CLASSES}, got {positive_class!r}")
    relations = set(df["relation_id"])
    if len(relations) > 1:
        raise MixedRelationError(f"labels of different relations are never pooled: {sorted(relations)}")
    outcome = df["outcome"].to_numpy(dtype=object)
    bad = set(outcome) - {o.value for o in Outcome}
    if bad:
        raise DataValidationError(f"metrics: unknown outcome(s) {sorted(bad)}")
    in_q = outcome != Outcome.NOT_APPLICABLE.value
    y_true = df["y_match"].to_numpy(dtype=float)[in_q]
    accepted = outcome[in_q] == Outcome.DECIDED.value
    y_pred = df["y_pred"].to_numpy(dtype=float, na_value=np.nan)[in_q]
    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred[accepted], (0, 1)).all():
        raise DataValidationError("metrics: y_match and y_pred of decided rows must be 0 or 1")
    if positive_class == "defect":   # y_defect = 1 − y_match, y_pred_defect = 1 − y_pred; rejects stay rejects
        y_true, y_pred = 1 - y_true, 1 - y_pred
    return y_true.astype(int), y_pred, accepted, int((~in_q).sum())


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    out = np.zeros((2, 3), dtype=int)
    for row, cls in enumerate((1, 0)):
        mask = y_true == cls
        out[row] = [(mask & accepted & (y_pred == 1)).sum(), (mask & accepted & (y_pred == 0)).sum(),
                    (mask & ~accepted).sum()]
    return out


def confusion_2x3(df: pd.DataFrame, positive_class: str) -> np.ndarray:
    """Rows [positive, negative]; columns [pred_pos, pred_neg, reject]."""
    return _confusion(*_labels(df, positive_class)[:3])


def compute_metrics(df: pd.DataFrame, positive_class: Literal["match", "defect"] = "match") -> MetricsRow:
    y_true, y_pred, accepted, n_na = _labels(df, positive_class)
    (tp, fn, rej_pos), (fp, tn, rej_neg) = _confusion(y_true, y_pred, accepted).tolist()
    n_pos, n_neg = tp + fn + rej_pos, fp + tn + rej_neg
    reasons: dict[str, str] = {}

    def ratio(name: str, num: int, den: int, reason: str) -> float | None:
        if den == 0:
            reasons[name] = reason
            return None
        return num / den

    coverage = ratio("coverage", tp + fn + fp + tn, n_pos + n_neg, EMPTY_CLASS)
    row = MetricsRow(
        tp=tp, fp=fp, tn=tn, fn=fn, rej_pos=rej_pos, rej_neg=rej_neg, n_pos=n_pos, n_neg=n_neg, n_not_applicable=n_na,
        coverage=coverage,
        coverage_pos=ratio("coverage_pos", tp + fn, n_pos, EMPTY_CLASS),
        coverage_neg=ratio("coverage_neg", fp + tn, n_neg, EMPTY_CLASS),
        rho_rej=None if coverage is None else 1 - coverage,
        recall_all=ratio("recall_all", tp, n_pos, EMPTY_CLASS),
        fpr_all=ratio("fpr_all", fp, n_neg, EMPTY_CLASS),
        correct_all=ratio("correct_all", tp + tn, n_pos + n_neg, EMPTY_CLASS),
        recall_cond=ratio("recall_cond", tp, tp + fn, NO_ACCEPTED),
        fpr_cond=ratio("fpr_cond", fp, fp + tn, NO_ACCEPTED),
        correct_cond=ratio("correct_cond", tp + tn, tp + tn + fp + fn, NO_ACCEPTED),
        positive_class=positive_class, null_reasons=reasons)
    if coverage is None:
        reasons["rho_rej"] = EMPTY_CLASS
    return row
