import numpy as np
import pandas as pd
import pytest

from metric.sampler import BatchSampler
from expcore.loaders import ImageStore, SplitGuard

pytestmark = pytest.mark.tf


def _store(root, arrays):
    import cv2
    for k, a in arrays.items():
        cv2.imencode(".png", a)[1].tofile(str(root / f"{k}.png"))
    df = pd.DataFrame({"image_id": list(arrays), "relative_path": [f"{k}.png" for k in arrays],
                       "split": "train"})
    return ImageStore(root, df, SplitGuard("smoke", fixture=True))


def test_dataset_order(tmp_path, tf_or_skip):
    from metric.pipeline import eval_arrays, train_dataset
    rng = np.random.default_rng(0)
    arrays, rows = {}, []
    for c in range(4):
        for k in range(3):
            iid = f"i{c}_{k}"
            h, w = 40 + 3 * k, 50 + 5 * c
            arrays[iid] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            rows.append((iid, str(c), w, h))
    store = _store(tmp_path, arrays)
    df = pd.DataFrame(rows, columns=["image_id", "class_id", "width", "height"])
    from metric.data import build_proxy_index
    s = BatchSampler(build_proxy_index(df), df, 1, 4, 2, 16, 8, True)

    def run():
        return list(train_dataset(s, store, 5, 2, 1).as_numpy_iterator())
    a, b = run(), run()
    for i, (x, lab, step) in enumerate(a):
        spec = s.batch(5 + i)
        assert x.shape == (8, 8, 8, 3) and x.dtype == np.float32
        assert lab.dtype == np.int32 and step.dtype == np.int64 and step == 5 + i
        np.testing.assert_array_equal(lab, spec.proxy_idx)
        np.testing.assert_array_equal(x, b[i][0])

    ids = list(arrays)
    out = list(eval_arrays(ids, store, 16, 8, 5, "noise_0.03", 1))
    assert [i for c, _ in out for i in c] == ids
    assert out[0][1].shape == (5, 8, 8, 3) and out[0][1].dtype == np.float32
