import pytest

from metric.augment import resized_shape

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("h,w,short,want", [
    (480, 640, 256, (256, 341)),
    (640, 480, 256, (341, 256)),
    (256, 256, 256, (256, 256)),
    (50, 300, 256, (256, 1536)),
    (2, 3, 1, (1, 2)),
    (100, 150, 256, (256, 384)),
])
def test_resized_shape_table(h, w, short, want):
    assert resized_shape(h, w, short) == want


import cv2
import numpy as np
import pandas as pd

from expcore.loaders import ImageStore, SplitGuard
from metric.augment import center_offsets, degrade, resize_short, to_model_input


def _tf_pre():
    pytest.importorskip("tensorflow")


def test_preprocess_once():
    _tf_pre()
    img = np.zeros((2, 2, 3), np.uint8)
    img[:] = (10, 20, 30)
    out = to_model_input(img)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [30 - 103.939, 20 - 116.779, 10 - 123.68], atol=1e-4)


def test_center_crop_and_flip():
    g = np.tile(np.arange(300) % 256 * 0 + np.arange(300) // 2, (256, 1))[..., None].repeat(3, 2).astype(np.uint8)
    y, x = center_offsets(256, 300, 224)
    assert (y, x) == (16, 38)
    c = g[y:y + 224, x:x + 224]
    assert c[0, 0, 0] == 19 and c[0, -1, 0] == 130
    assert c[:, ::-1][0, 0, 0] == 130


def _write(root, name, arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    buf.tofile(str(root / name))


def _store(tmp_path, arrays):
    for k, a in arrays.items():
        _write(tmp_path, f"{k}.png", a)
    df = pd.DataFrame({"image_id": list(arrays), "relative_path": [f"{k}.png" for k in arrays],
                       "split": "train"})
    return ImageStore(tmp_path, df, SplitGuard("smoke", fixture=True))


def test_odd_images(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {
        "wide": rng.integers(0, 255, (50, 300, 3), dtype=np.uint8),
        "tall": rng.integers(0, 255, (300, 50, 3), dtype=np.uint8),
        "gray": rng.integers(0, 255, (60, 80), dtype=np.uint8),
        "rgba": rng.integers(0, 255, (60, 80, 4), dtype=np.uint8),
    }
    store = _store(tmp_path, arrays)
    for k in arrays:
        img = store.load(k, color="rgb")
        assert img.ndim == 3 and img.shape[2] == 3
        r = resize_short(img, 256)
        assert r.shape[:2] == resized_shape(*img.shape[:2], 256)
        y, x = center_offsets(*r.shape[:2], 224)
        assert r[y:y + 224, x:x + 224].shape == (224, 224, 3)


def test_degrade(rng):
    assert degrade(np.full((2, 2, 3), 0.95), "bright_+0.1", None).max() == 1.0
    assert degrade(np.full((2, 2, 3), 0.05), "bright_-0.1", None).min() == 0.0
    x = np.full((200, 200, 3), 0.5)
    assert degrade(x, "none", None) is x
    a = degrade(x, "noise_0.03", np.random.default_rng(1))
    b = degrade(x, "noise_0.03", np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert np.std(a - x) == pytest.approx(0.03, rel=0.1)
    with pytest.raises(ValueError):
        degrade(x, "bogus", None)
