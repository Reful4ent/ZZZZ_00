import pandas as pd
import pytest

from expcore.errors import LeakageError
from metric.data import (apply_class_subset, apply_per_class_limit, build_proxy_index,
                         check_disjoint, sampler_exclusions, steps_per_epoch)

pytestmark = pytest.mark.unit


def frame(spec, split="train"):
    rows = [(f"i{c}_{k}_{split}", str(c), split) for c, n in spec.items() for k in range(n)]
    return pd.DataFrame(rows, columns=["image_id", "class_id", "split"])


def test_disjoint_raises():
    with pytest.raises(LeakageError):
        check_disjoint(pd.concat([frame({1: 2}), frame({1: 2}, "test")]))
    check_disjoint(pd.concat([frame({1: 2}), frame({2: 2}, "test")]))


def test_proxy_index_train_only():
    pi = build_proxy_index(frame({10: 2, 2: 3, 1: 2}))
    assert list(pi["class_id"]) == ["1", "2", "10"]
    assert list(pi["proxy_idx"]) == [0, 1, 2]


def test_exclusions():
    pi = build_proxy_index(frame({1: 1, 2: 3}))
    assert list(pi["sampler_eligible"]) == [False, True]
    ex = sampler_exclusions(pi)
    assert list(ex["class_id"]) == ["1"] and list(ex["n_images"]) == [1]
    assert list(ex["reason"]) == ["FEWER_THAN_2_IMAGES"]


def test_class_subset():
    df = frame({c: 2 for c in range(20)})
    a = apply_class_subset(df, 5, 7)
    assert a["class_id"].nunique() == 5
    assert a.equals(apply_class_subset(df, 5, 7))


def test_per_class_limit_nested():
    df = frame({1: 6, 2: 4, 3: 3})
    sets = {m: apply_per_class_limit(df, m, 4) for m in (2, 3, 4, None)}
    assert set(sets[None]["class_id"]) == {"1", "2"}
    prev = None
    for m in (2, 3, 4, None):
        cur = {c: set(g["image_id"]) for c, g in sets[m].groupby("class_id")}
        if m:
            assert all(len(v) == m for v in cur.values())
        if prev:
            assert all(prev[c] <= cur[c] for c in cur)
        prev = cur


def test_steps_per_epoch():
    assert steps_per_epoch(47616) == 1488
    assert steps_per_epoch(47617) == 1489
    assert steps_per_epoch(1) == 1


def test_class_subset_independent_of_run_seed():   # spec 4.5: subset seed comes from params, not the run
    df = pd.DataFrame({"class_id": [str(i % 12) for i in range(48)], "image_id": range(48)})
    cs = {"n_classes": 5, "seed": 3}
    sets = [set(apply_class_subset(df, cs["n_classes"], cs["seed"])["class_id"]) for _run_seed in (11, 23, 37)]
    assert sets[0] == sets[1] == sets[2] and len(sets[0]) == 5
