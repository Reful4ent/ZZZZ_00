"""Thresholds are chosen on split=validation only (spec §6.4, INV-10) and then read back read-only."""

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from expcore.errors import DataValidationError, InsufficientDataError, ProtocolViolationError, ThresholdLeakageError
from expcore.io import dump_json, load_json

VALIDATION = "validation"


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
    """`criterion(unified_scores, y_match)` → threshold on the unified scale; rows without a score are skipped."""
    leaked = sorted(set(val["split"]) - {VALIDATION})
    if leaked:
        raise ThresholdLeakageError(f"thresholds may be chosen on split=validation only, got {leaked} (INV-10)")
    if val.empty:
        raise InsufficientDataError("select_threshold: no validation pairs")
    scored = val[val["score_raw"].notna()]
    raw = scored["score_raw"].to_numpy(dtype=float)
    unified = np.where(scored["score_direction"].to_numpy() == "lower_is_match", -raw, raw)
    value = float(criterion(unified, scored["y_match"].to_numpy(dtype=int)))
    if not math.isfinite(value):
        raise DataValidationError(f"criterion {criterion_id!r} returned a non-finite threshold {value}")
    return Threshold(run_id, method_id, criterion_id, value, VALIDATION, len(val), manifest_hash)


def save_threshold(t: Threshold, path: Path) -> None:
    if Path(path).exists():
        raise ProtocolViolationError(f"{path}: thresholds are written once and then only read")
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
