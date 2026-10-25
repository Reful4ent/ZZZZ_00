"""ПРИЁМ-Ф4: детерминизм манифеста, одинаковые пары и разбиение для всех методов."""

import pandas as pd
import pytest

from photometric.pairs import build_pairs, manifest_digest
from photometric.split import split_originals


def _setup(n=20):
    ids = [f"I{i:02d}" for i in range(1, n + 1)]
    originals = pd.DataFrame({"original_id": ids, "path": [f"/x/{i}.png" for i in ids]})
    queries = pd.DataFrame(
        [{"query_id": f"{o}_{t}_{l:02d}", "query_path": f"/x/{o}_{t}_{l:02d}.png", "original_id": o, "type_id": t, "level": l}
         for o in ids for t in ("07", "10") for l in range(1, 6)]
    )
    return ids, originals, queries


def test_split_deterministic_and_disjoint():
    ids, *_ = _setup(81)
    s1 = split_originals(ids, 7, {"validation": 0.5, "test": 0.5})
    s2 = split_originals(list(reversed(ids)), 7, {"validation": 0.5, "test": 0.5})
    assert s1 == s2
    assert set(s1) == set(ids)
    assert sum(v == "validation" for v in s1.values()) == 40
    assert s1 != split_originals(ids, 8, {"validation": 0.5, "test": 0.5})


def test_manifest_same_seed_same_manifest():
    ids, originals, queries = _setup()
    split_of = split_originals(ids, 1, {"validation": 0.5, "test": 0.5})
    m1 = build_pairs(queries, originals, split_of, 42, "P1-E1")
    m2 = build_pairs(queries.sample(frac=1, random_state=0), originals, split_of, 42, "P1-E1")
    pd.testing.assert_frame_equal(m1, m2)
    assert manifest_digest(m1) == manifest_digest(m2)
    m3 = build_pairs(queries, originals, split_of, 43, "P1-E1")
    assert manifest_digest(m1) != manifest_digest(m3)


def test_pair_rules():
    ids, originals, queries = _setup()
    split_of = split_originals(ids, 1, {"validation": 0.5, "test": 0.5})
    m = build_pairs(queries, originals, split_of, 42, "P1-E1")
    assert len(m) == 4 * len(queries)
    for qid, g in m.groupby("query_id"):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]
        assert len(pos) == 1 and pos["reference_id"].iloc[0] == g["original_id"].iloc[0]
        assert len(neg) == 3
        assert neg["reference_id"].nunique() == 3
        assert (neg["reference_id"] != g["original_id"].iloc[0]).all()
        # отрицательные — из той же части
        assert {split_of[r] for r in neg["reference_id"]} == {split_of[g["original_id"].iloc[0]]}
        assert (g["split"] == split_of[g["original_id"].iloc[0]]).all()


def test_too_small_part_rejected():
    ids, originals, queries = _setup(5)
    split_of = {o: ("validation" if i < 2 else "test") for i, o in enumerate(ids)}
    with pytest.raises(ValueError):
        build_pairs(queries, originals, split_of, 1, "P1-E1")
