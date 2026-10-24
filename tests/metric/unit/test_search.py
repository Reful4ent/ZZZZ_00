import numpy as np
import pytest

from metric.errors import RunStop
from metric.search import (
    SearchResult,
    blocked_topk,
    check_and_clip_d2,
    no_positive_mask,
    retrieval_table,
)

pytestmark = pytest.mark.unit


def _data(n=50, d=8, ncls=5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    ids = np.array([f"img{i:03d}" for i in range(n)])
    cls = np.array([f"c{i % ncls}" for i in range(n)])
    return z, ids, cls


def _oracle(zq, q_ids, zg, g_ids, k):
    d2 = ((zq[:, None, :] - zg[None, :, :]) ** 2).sum(-1)
    d2[q_ids[:, None] == g_ids[None, :]] = np.inf
    out = []
    for i in range(len(zq)):
        order = np.lexsort((g_ids, d2[i]))[:k]
        out.append([j if np.isfinite(d2[i, j]) else -1 for j in order])
    return np.array(out)


@pytest.mark.parametrize("bq,bg", [(1, 1), (7, 13), (64, 64)])
def test_blocked_equals_bruteforce(bq, bg):
    z, ids, _ = _data()
    res = blocked_topk(z, ids, z, ids, 10, bq, bg, 1e-6, "cosine")
    assert np.array_equal(res.top_idx, _oracle(z, ids, z, ids, 10))


def test_no_self_match():
    z, ids, _ = _data()
    res = blocked_topk(z, ids, z, ids, 5, 7, 13, 1e-6, "cosine")
    for i in range(len(ids)):
        assert ids[i] not in ids[res.top_idx[i]]


def test_tie_break_by_image_id():
    zg = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    g_ids = np.array(["b", "a", "z"])
    res = blocked_topk(np.array([[1.0, 0.0]]), np.array(["q"]), zg, g_ids, 2, 1, 1, 1e-6, "cosine")
    assert list(g_ids[res.top_idx[0]]) == ["a", "b"]


def test_cosine_equals_sql2():
    # axis-aligned vectors: many exactly equal distances
    zg = np.vstack([np.eye(4), -np.eye(4)])
    ids = np.array([f"g{i}" for i in range(8)])
    zq = np.eye(4)
    q_ids = np.array([f"q{i}" for i in range(4)])
    res = blocked_topk(zq, q_ids, zg, ids, 8, 2, 3, 1e-6, "cosine")
    assert np.array_equal(res.top_idx, _oracle(zq, q_ids, zg, ids, 8))


def test_raw_l2_matches_oracle():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(20, 4)) * 3
    ids = np.array([f"i{i:02d}" for i in range(20)])
    res = blocked_topk(z, ids, z, ids, 5, 6, 7, 1e-6, "raw_l2")
    assert np.array_equal(res.top_idx, _oracle(z, ids, z, ids, 5))


def test_range_violation():
    assert np.array_equal(check_and_clip_d2(np.array([-1e-7, 4 + 5e-7]), 1e-6), [0.0, 4.0])
    with pytest.raises(RunStop) as e:
        check_and_clip_d2(np.array([-1e-3]), 1e-6)
    assert e.value.failure_type == "DISTANCE_RANGE_VIOLATION"


def test_no_positive_excluded():
    q_ids = np.array(["a", "b", "c"])
    q_cls = np.array(["x", "x", "y"])
    mask = no_positive_mask(q_ids, q_cls, q_ids, q_cls)
    assert list(mask) == [False, False, True]
    res = SearchResult(q_ids, np.array([[1, 2], [0, 2], [0, 1]]), np.array([[1.0, 2], [1, 2], [1, 2]]))
    df, s = retrieval_table(res, q_cls, q_ids, q_cls, mask)
    row = df.iloc[2]
    assert not row["valid"] and row["reason_code"] == "NOT_APPLICABLE_NO_POSITIVE_IN_GALLERY"
    assert s["n_no_positive"] == 1 and s["n_valid"] == 2 and s["recall_at_1"] == 1.0


def test_recall_at_k_values():
    g_ids = np.array(["g0", "g1", "g2", "g3", "g4"])
    g_cls = np.array(["A", "B", "A", "B", "A"])
    q_cls = np.array(["A", "A", "B", "B"])
    top = np.array([[0, 1, 3], [1, 3, 0], [3, 0, 1], [0, 2, 1]])
    res = SearchResult(np.array(["q0", "q1", "q2", "q3"]), top, np.tile(np.arange(3.0), (4, 1)) ** 2)
    df, s = retrieval_table(res, q_cls, g_ids, g_cls, np.zeros(4, bool))
    assert s["recall_at_1"] == 0.5 and s["recall_at_5"] == 1.0 and s["correct_at_1"] == 2
    assert df["rank_first_positive"].tolist() == [1, 3, 1, 3]


def test_small_gallery_padding():
    z, ids, cls = _data(n=3, ncls=1)
    res = blocked_topk(z, ids, z, ids, 10, 2, 2, 1e-6, "cosine")
    assert (res.top_idx[:, 2:] == -1).all() and np.isinf(res.top_d2[:, 2:]).all()
    df, s = retrieval_table(res, cls, ids, cls, no_positive_mask(ids, cls, ids, cls))
    assert s["recall_at_10"] == 1.0
    assert len(df.iloc[0]["top_ids"]) == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_range_violation_nan(bad):
    with pytest.raises(RunStop) as e:
        check_and_clip_d2(np.array([bad, 1.0]), 1e-6)
    assert e.value.failure_type == "DISTANCE_RANGE_VIOLATION"


@pytest.mark.parametrize("metric", ["cosine", "raw_l2"])
def test_blocked_topk_nan_embedding_raises(metric):
    z, ids, _ = _data(n=10)
    z[3] = np.nan
    with pytest.raises(RunStop) as e:
        blocked_topk(z, ids, z, ids, 5, 4, 4, 1e-6, metric)
    assert e.value.failure_type == "DISTANCE_RANGE_VIOLATION"
