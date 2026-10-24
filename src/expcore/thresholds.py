"""Thresholds are chosen on split=validation only (spec §6.4, INV-10) and then read back read-only."""

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from expcore.decisions import ScoreDirection
from expcore.errors import (DataValidationError, InsufficientDataError, ProtocolViolationError, ResultTamperedError,
                            ThresholdLeakageError)
from expcore.io import dump_json, load_json
from expcore.registry import Outcome

VALIDATION = "validation"
_REQUIRED = ("split", "y_match", "score_raw", "score_direction", "outcome")


@dataclass(frozen=True)
class Threshold:
    run_id: str
    method_id: str
    criterion_id: str
    value: float
    split_used: str
    n_validation_pairs: int
    manifest_hash: str


def select_threshold(val: pd.DataFrame, criterion: Callable[[np.ndarray, np.ndarray], float], *, run_id: str,
                     method_id: str, criterion_id: str, manifest_hash: str) -> Threshold:
    """`criterion(unified_scores, y_match)` → threshold on the unified scale.

    Only decided rows with a score reach the criterion: not_applicable is outside Q (§5.7) and a
    reject is not a decision, whatever score it carries.
    """
    missing = [c for c in _REQUIRED if c not in val.columns]
    if missing:
        raise DataValidationError(f"select_threshold: missing columns {missing}")
    leaked = sorted(set(val["split"]) - {VALIDATION})
    if leaked:
        raise ThresholdLeakageError(f"thresholds may be chosen on split=validation only, got {leaked} (INV-10)")
    if val.empty:
        raise InsufficientDataError("select_threshold: no validation pairs")
    bad = set(val["outcome"]) - {o.value for o in Outcome}
    if bad:
        raise DataValidationError(f"select_threshold: unknown outcome(s) {sorted(bad)}")
    if not val["y_match"].isin([0, 1]).all():
        raise DataValidationError("select_threshold: y_match must be 0 or 1 in every row")
    try:
        negate = np.array([ScoreDirection(d) is ScoreDirection.LOWER_IS_MATCH for d in val["score_direction"]],
                          dtype=bool)
    except ValueError as e:
        raise DataValidationError(f"select_threshold: {e}") from e
    raw = val["score_raw"].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(raw).any():
        raise DataValidationError("select_threshold: score_raw contains Inf")
    used = (val["outcome"].to_numpy() == Outcome.DECIDED.value) & ~np.isnan(raw)
    unified = np.where(negate, -raw, raw)[used]
    value = float(criterion(unified, val["y_match"].to_numpy(dtype=int)[used]))
    if not math.isfinite(value):
        raise DataValidationError(f"criterion {criterion_id!r} returned a non-finite threshold {value}")
    return Threshold(run_id, method_id, criterion_id, value, VALIDATION, len(val), manifest_hash)


def save_threshold(t: Threshold, path: Path) -> None:
    """Written once; saving the same threshold again (resume, re-aggregate) is a no-op, a different one is refused."""
    if Path(path).exists():
        if load_threshold(path) != t:
            raise ResultTamperedError(f"{path}: a different threshold is already recorded; thresholds are never "
                                      "rewritten")
        return
    dump_json({**dataclasses.asdict(t), "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
              path)


def load_threshold(path: Path) -> Threshold:
    if not Path(path).is_file():
        raise ProtocolViolationError(f"{path}: thresholds.json is missing; choose the threshold on validation first")
    data = load_json(path)
    names = [f.name for f in dataclasses.fields(Threshold)]
    if not isinstance(data, dict) or set(data) != set(names) | {"created_at"}:
        raise DataValidationError(f"{path}: expected keys {names + ['created_at']}")
    if data["split_used"] != VALIDATION:
        raise ProtocolViolationError(f"{path}: split_used is {data['split_used']!r}, not validation (INV-10)")
    return Threshold(**{n: data[n] for n in names})
