import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from metric.augment import resized_shape
from metric.data import build_proxy_index
from metric.errors import RunStop
from metric.sampler import BatchSampler

pytestmark = pytest.mark.unit


def imgs(n_classes, per=3, singles=()):
    rows = []
    for c in range(n_classes):
        for k in range(1 if c in singles else per):
            rows.append((f"i{c}_{k}", str(c), 300 + 7 * k, 400 - 5 * c))
    return pd.DataFrame(rows, columns=["image_id", "class_id", "width", "height"])


def make(df, seed=1, aug=True, classes=16):
    return BatchSampler(build_proxy_index(df), df, seed, classes, 2, 256, 224, aug)


def test_batch_shape_rules():
    s = make(imgs(40, singles=(3, 5)))
    for t in range(1000):
        b = s.batch(t)
        assert len(set(b.class_ids)) == 16 and len(set(b.image_ids)) == 32
        assert "3" not in b.class_ids and "5" not in b.class_ids
        for i in range(16):
            assert b.class_ids[2 * i] == b.class_ids[2 * i + 1]
            assert b.image_ids[2 * i] != b.image_ids[2 * i + 1]
    assert b.proxy_idx.dtype == np.int32 and b.crop_yx.shape == (32, 2)


def test_uniform_classes():
    s = make(imgs(40))
    counts = np.zeros(40)
    for t in range(10_000):
        for c in s.batch(t).class_ids:
            counts[int(c)] += 1
    assert chisquare(counts).pvalue > 0.001


def test_batch_pure_function():
    df = imgs(40)
    a, b = make(df).batch(17), make(df).batch(17)
    assert a.image_ids == b.image_ids and np.array_equal(a.crop_yx, b.crop_yx)
    assert make(df, seed=2).batch(17).image_ids != a.image_ids


def test_crop_offsets_valid():
    df = imgs(40)
    s = make(df)
    for t in range(50):
        b = s.batch(t)
        for iid, (y, x) in zip(b.image_ids, b.crop_yx):
            h, w = resized_shape(df.loc[df.image_id == iid, "height"].item(), df.loc[df.image_id == iid, "width"].item(), 256)
            assert min(h, w) == 256 and 0 <= y <= h - 224 and 0 <= x <= w - 224
    sc = make(df, aug=False)
    c = sc.batch(3)
    assert not c.flip.any()
    for iid, (y, x) in zip(c.image_ids, c.crop_yx):
        h, w = resized_shape(df.loc[df.image_id == iid, "height"].item(), df.loc[df.image_id == iid, "width"].item(), 256)
        assert (y, x) == ((h - 224) // 2, (w - 224) // 2)


def test_too_few_classes():
    with pytest.raises(RunStop) as e:
        make(imgs(10))
    assert e.value.failure_type == "SAMPLER_ERROR"


def test_to_frame():
    s = make(imgs(40))
    f = s.to_frame(s.batch(4))
    assert list(f.columns) == ["global_step", "pos", "image_id", "class_id", "proxy_idx",
                               "crop_y", "crop_x", "flip"]
    assert len(f) == 32 and (f["global_step"] == 4).all()
