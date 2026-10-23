"""Strict JSON (no NaN/Inf), atomic file writes and parquet I/O."""

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from expcore.errors import DataValidationError


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def dump_json(obj: Any, path: Path) -> None:
    try:
        text = json.dumps(obj, allow_nan=False, ensure_ascii=False, indent=2)
    except ValueError as e:
        raise DataValidationError(f"{path}: value not representable in strict JSON: {e}") from e
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    for col in df.select_dtypes(include="floating").columns:
        if not np.isfinite(df[col].to_numpy()).all():
            raise DataValidationError(f"{path}: column {col!r} contains NaN/Inf")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    atomic_write_bytes(path, buf.getvalue())


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
