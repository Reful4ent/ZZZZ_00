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


def _noise_setup(tmp_path):
    import cv2  # noqa: F401
    rng = np.random.default_rng(3)
    arrays, rows = {}, []
    for c in range(4):
        for k in range(3):
            iid = f"i{c}_{k}"
            arrays[iid] = rng.integers(0, 255, (300 + 10 * k, 400 + 20 * c, 3), dtype=np.uint8)
            rows.append((iid, str(c), arrays[iid].shape[1], arrays[iid].shape[0]))
    df = pd.DataFrame(rows, columns=["image_id", "class_id", "width", "height"])
    return arrays, df, _store(tmp_path, arrays)


def test_train_crop_flip_pixels(tmp_path, tf_or_skip):
    import cv2
    from metric.augment import resized_shape, to_model_input
    from metric.data import build_proxy_index
    from metric.pipeline import train_dataset
    arrays, df, store = _noise_setup(tmp_path)
    s = BatchSampler(build_proxy_index(df), df, 1, 4, 2, 64, 48, True)
    spec = s.batch(0)
    assert spec.flip.any() and (~spec.flip).any()
    x = next(iter(train_dataset(s, store, 0, 1, 1).as_numpy_iterator()))[0]
    for j, iid in enumerate(spec.image_ids):
        img = arrays[iid][..., ::-1]  # imencode wrote it as BGR; store returns RGB
        h, w = resized_shape(*img.shape[:2], 64)
        y0, x0 = spec.crop_yx[j]
        ref = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y0 + 48, x0:x0 + 48]
        if spec.flip[j]:
            ref = ref[:, ::-1]
        np.testing.assert_allclose(x[j], to_model_input(ref), atol=1e-4)


def test_eval_arrays_degradations(tmp_path, tf_or_skip):
    import cv2
    from metric.augment import resized_shape, to_model_input
    from metric.pipeline import eval_arrays
    arrays, _, store = _noise_setup(tmp_path)
    ids = list(arrays)[:6]

    def run(deg, seed, batch):
        return np.concatenate([x for _, x in eval_arrays(ids, store, 64, 48, batch, deg, seed)])
    refs = []
    for iid in ids:
        img = arrays[iid][..., ::-1]  # imencode wrote it as BGR; store returns RGB
        h, w = resized_shape(*img.shape[:2], 64)
        r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (h - 48) // 2, (w - 48) // 2
        refs.append(to_model_input(r[y0:y0 + 48, x0:x0 + 48]))
    none = run("none", 1, 4)
    np.testing.assert_allclose(none, np.stack(refs), atol=1e-4)
    a = run("noise_0.03", 1, 2)
    np.testing.assert_array_equal(a, run("noise_0.03", 1, 3))
    assert not np.array_equal(a, run("noise_0.03", 2, 2))
    assert not np.array_equal(a, none)


def test_noise_uses_degrade_stream(tmp_path):
    from expcore.seeding import stream
    from metric.augment import degrade
    from metric.pipeline import _crop, eval_arrays
    arrays, _, store = _noise_setup(tmp_path)
    ids = list(arrays)[:3]
    x = np.concatenate([b for _, b in eval_arrays(ids, store, 64, 48, 2, "noise_0.03", 7, raw=True)])
    for i, iid in enumerate(ids):
        ref = degrade(_crop(store, iid, 64, 48) / 255.0, "noise_0.03", stream(7, "degrade", i)) * 255.0
        np.testing.assert_allclose(x[i], ref.astype(np.float32), atol=1e-4)
