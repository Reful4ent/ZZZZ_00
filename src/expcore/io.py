"""Strict JSON (no NaN/Inf), atomic file writes and parquet I/O."""

import io
import json
import os
from pathlib import Path
from typing import Any, Collection, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

from expcore.errors import DataValidationError

# Contract dtype names (spec §5) -> Arrow types.
_ARROW_TYPES = {
    "str": pa.string(), "bool": pa.bool_(), "int8": pa.int8(), "int32": pa.int32(),
    "int64": pa.int64(), "float64": pa.float64(),
}


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
    def reject_constant(name: str) -> Any:
        raise DataValidationError(f"{path}: {name} is not allowed in strict JSON")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict:
        keys = [k for k, _ in pairs]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise DataValidationError(f"{path}: duplicate keys {dup}")
        return dict(pairs)

    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def write_parquet(df: pd.DataFrame, path: Path, *, nullable: Collection[str] = (),
                  schema: Mapping[str, str] | None = None) -> None:
    """Write atomically. Inf is always rejected; NaN only in `nullable` columns, stored as null.

    `schema` (column -> contract dtype) fixes the Arrow types, so all-null columns keep their type.
    """
    arrow_schema = None
    if schema is not None:
        arrow_schema = pa.schema([(c, _ARROW_TYPES[schema[c]]) for c in df.columns])
    try:
        table = pa.Table.from_pandas(df, schema=arrow_schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        raise DataValidationError(f"{path}: table does not match its schema: {e}") from e
    # from_pandas turns NaN into null, so a null in a float column was either None or NaN.
    for name, col in zip(table.column_names, table.columns):
        if not pa.types.is_floating(col.type):
            continue
        if col.null_count and name not in nullable:
            raise DataValidationError(f"{path}: column {name!r} contains NaN/null")
        if pc.all(pc.is_finite(col)).as_py() is False:  # nulls are skipped
            raise DataValidationError(f"{path}: column {name!r} contains Inf")
    buf = io.BytesIO()
    pq.write_table(table, buf)
    atomic_write_bytes(path, buf.getvalue())


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
