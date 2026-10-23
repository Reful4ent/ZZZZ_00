import os

import numpy as np
import pandas as pd
import pytest

from expcore.errors import DataValidationError
from expcore.hashing import canonical_json, sha256_file, table_content_hash
from expcore.io import atomic_write_bytes, dump_json, load_json, read_parquet, write_parquet

pytestmark = pytest.mark.unit


def test_sha256_known(tmp_path):
    p = tmp_path/"a"; p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_dump_json_rejects_nan(tmp_path):
    with pytest.raises(DataValidationError): dump_json({"x": float("nan")}, tmp_path/"x.json")
    assert not (tmp_path/"x.json").exists()


def test_parquet_rejects_inf(tmp_path):
    with pytest.raises(DataValidationError): write_parquet(pd.DataFrame({"s": [1.0, np.inf]}), tmp_path/"p.parquet")


def test_table_hash_order_and_roundtrip(tmp_path):   # T-DAT-09
    df = pd.DataFrame({"k": ["b", "a"], "v": [2, 1]})
    h = table_content_hash(df, ["k", "v"], ["k"])
    assert h == table_content_hash(df.iloc[::-1], ["k", "v"], ["k"])
    write_parquet(df, tmp_path/"t.parquet")
    assert h == table_content_hash(read_parquet(tmp_path/"t.parquet"), ["k", "v"], ["k"])


def test_atomic_write_failure_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError): atomic_write_bytes(tmp_path/"f", b"x")
    assert list(tmp_path.iterdir()) == []


def test_parquet_nullable_float_columns(tmp_path):
    df = pd.DataFrame({"s": [1.0, None], "t": [1.0, 2.0]})
    with pytest.raises(DataValidationError): write_parquet(df, tmp_path/"p.parquet")
    write_parquet(df, tmp_path/"p.parquet", nullable=("s",))
    assert read_parquet(tmp_path/"p.parquet")["s"].isna().tolist() == [False, True]
    with pytest.raises(DataValidationError):
        write_parquet(pd.DataFrame({"s": [np.inf]}), tmp_path/"q.parquet", nullable=("s",))


def test_parquet_object_column_inf_rejected(tmp_path):   # review: INV-08 bypass via object dtype
    df = pd.concat([pd.DataFrame({"s": [None]}), pd.DataFrame({"s": [np.inf]})], ignore_index=True)
    assert df["s"].dtype == object
    with pytest.raises(DataValidationError): write_parquet(df, tmp_path/"p.parquet", nullable=("s",))


def test_parquet_schema_types_all_null_columns(tmp_path):   # review: all-reject shard got Arrow null type
    import pyarrow.parquet as pq
    df = pd.DataFrame({"s": [None, None], "y": [None, None], "k": ["a", "b"]})
    write_parquet(df, tmp_path/"p.parquet", nullable=("s", "y"), schema={"s": "float64", "y": "int8", "k": "str"})
    types = {f.name: str(f.type) for f in pq.read_schema(tmp_path/"p.parquet")}
    assert types == {"s": "double", "y": "int8", "k": "string"}


def test_table_hash_nullable_int_roundtrip(tmp_path):   # review: D-07 broke for nullable int columns
    df = pd.DataFrame({"k": ["a", "b"], "seed": [2026, None]}, dtype=object)
    h = table_content_hash(df, ["k", "seed"], ["k"])
    write_parquet(df, tmp_path/"t.parquet", nullable=("seed",), schema={"k": "str", "seed": "int64"})
    assert h == table_content_hash(read_parquet(tmp_path/"t.parquet"), ["k", "seed"], ["k"])


def test_load_json_strict(tmp_path):   # review: NaN/Infinity and duplicate keys were accepted on read
    for text in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
        (tmp_path/"a.json").write_text(text, encoding="utf-8")
        with pytest.raises(DataValidationError): load_json(tmp_path/"a.json")
