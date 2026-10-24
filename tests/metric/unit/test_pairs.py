import inspect

import numpy as np
import pandas as pd
import pytest

from expcore.schemas import PAIRS_COLUMNS, validate_pairs
from metric.pairs import build_p3_pairs, p3_pair_builder, pairs_summary, score_pairs

pytestmark = pytest.mark.unit


def make_images(spec, split="test"):
    """spec: {(class_id, category_id): n_images}"""
    rows = [
        {"image_id": f"sop/{c}/{i}", "dataset_id": "sop", "source_group_id": c, "class_id": c,
         "category_id": cat, "split": split, "relative_path": f"{c}_{i}.jpg", "sha256": "0" * 64,
         "width": 8, "height": 8, "channels": 3, "defect_type": None}
        for (c, cat), n in spec.items() for i in range(n)
    ]
    return pd.DataFrame(rows)


def rich_images():
    spec = {(str(c), str(c % 3)): 3 + c % 7 for c in range(1, 25)}
    return make_images(spec)


def test_pos_count_and_unique():
    imgs = make_images({("1", "a"): 3, ("2", "a"): 8, ("3", "b"): 4, ("4", "b"): 4})
    pairs, _ = build_p3_pairs(imgs, "test")
    pos = pairs[pairs.y_match == 1]
    assert (pos.source_group_id == "1").sum() == 3
    assert (pos.source_group_id == "2").sum() == 20
    keys = [tuple(sorted(t)) for t in zip(pairs.query_id, pairs.reference_id)]
    assert len(keys) == len(set(keys))
    assert list(pairs.columns) == list(PAIRS_COLUMNS)


def test_neg_split_by_superclass():
    spec = {("1", "a"): 8}
    spec.update({(str(c), "a"): 8 for c in range(2, 8)})
    spec.update({(str(c), "b"): 8 for c in range(8, 14)})
    pairs, summ = build_p3_pairs(make_images(spec), "test")
    cat = {"1": "a"}
    sub = pairs[(pairs.source_group_id == "1") & (pairs.y_match == 0)]
    assert len(sub) == 20
    same = sub.reference_id.str.split("/").str[1].map(lambda c: int(c) < 8)
    assert same.sum() == 10
    assert summ["frac_same_superclass"] == pytest.approx(
        summ["n_neg_same_superclass"] / summ["n_neg"])


def test_superclass_shortage_recorded():
    imgs = make_images({("1", "a"): 8, ("2", "b"): 8, ("3", "b"): 8})
    pairs, summ = build_p3_pairs(imgs, "test")
    assert summ["classes_with_shortage"] >= 1
    sub = pairs[(pairs.source_group_id == "1") & (pairs.y_match == 0)]
    assert len(sub) > 0 and sub.reference_id.str.split("/").str[1].isin(["2", "3"]).all()


def test_single_image_class_has_no_pairs():
    pairs, _ = build_p3_pairs(make_images({("1", "a"): 1, ("2", "a"): 3}), "test")
    assert "1" not in set(pairs.source_group_id)


def test_deterministic():
    a, sa = build_p3_pairs(rich_images(), "test")
    b, sb = build_p3_pairs(rich_images(), "test")
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb


def test_no_embedding_input():
    assert list(inspect.signature(build_p3_pairs).parameters) == [
        "images", "split", "seed", "max_pos", "max_attempts"]


def test_score_pairs():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    ids = np.array(["a", "b", "c"])
    pairs = pd.DataFrame({"pair_id": ["p1", "p2"], "query_id": ["a", "a"], "reference_id": ["b", "c"]})
    out = score_pairs(emb, ids, pairs, tol=1e-6)
    assert list(out.columns) == ["pair_id", "query_id", "reference_id", "s_cos", "d2", "d"]
    np.testing.assert_allclose(out.s_cos, [0.0, 0.6])
    np.testing.assert_allclose(out.d2, 2 - 2 * out.s_cos)
    np.testing.assert_allclose(out.d, np.sqrt(out.d2))


def test_builder_contract():
    val = rich_images().assign(split="validation", image_id=lambda d: d.image_id + "v",
                               class_id=lambda d: d.class_id + "0", source_group_id=lambda d: d.class_id)
    imgs = pd.concat([rich_images(), val], ignore_index=True)
    pairs = p3_pair_builder(imgs, None)
    validate_pairs(pairs, imgs)
    assert set(pairs.split) == {"validation", "test"}
    summ = pairs_summary(pairs, imgs)
    for split in ("validation", "test"):
        _, expected = build_p3_pairs(imgs[imgs.split == split], split)
        got = summ[split]
        assert {k: v for k, v in got.items() if v is not None} == {
            k: v for k, v in expected.items() if k not in ("classes_with_shortage", "skipped_pairs")}
    empty = p3_pair_builder(imgs.assign(dataset_id="kadid10k"), None)
    assert list(empty.columns) == list(PAIRS_COLUMNS) and empty.empty
