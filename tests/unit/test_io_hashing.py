import os

import numpy as np
import pandas as pd
import pytest

from expcore.errors import DataValidationError
from expcore.hashing import canonical_json, sha256_file, table_content_hash
from expcore.io import atomic_write_bytes, dump_json, read_parquet, write_parquet

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
