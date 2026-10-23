"""SHA-256 helpers and canonical serialization (D-07)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_BLOCK = 1 << 20


def _to_builtin(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_to_builtin)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_BLOCK):
            h.update(block)
    return h.hexdigest()


def table_content_hash(df: pd.DataFrame, columns: Sequence[str], key: Sequence[str]) -> str:
    """Hash of table content, independent of row order and parquet bytes (D-07)."""
    columns = list(columns)
    rows = df.sort_values(list(key), kind="mergesort")[columns].astype(object)
    rows = rows.where(rows.notna(), None)
    h = hashlib.sha256(canonical_json(columns).encode("utf-8"))
    for row in rows.itertuples(index=False, name=None):
        h.update(b"\n" + canonical_json(list(row)).encode("utf-8"))
    return h.hexdigest()
