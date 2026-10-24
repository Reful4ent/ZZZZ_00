"""Train tf.data pipeline and eval array iterator."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from expcore.seeding import stream
from metric.augment import center_offsets, degrade, resize_short, to_model_input


def _crop(store, iid, short, crop, yx=None, flip=False):
    img = resize_short(store.load(iid, color="rgb"), short)
    y, x = center_offsets(*img.shape[:2], crop) if yx is None else yx
    c = img[y:y + crop, x:x + crop]
    return c[:, ::-1] if flip else c


def train_dataset(sampler, store, start_step: int, n_steps: int, prefetch: int):
    import tensorflow as tf
    short, crop = sampler.resize_short, sampler.crop

    def gen():
        with ThreadPoolExecutor(8) as pool:
            for t in range(start_step, start_step + n_steps):
                b = sampler.batch(t)
                rows = pool.map(lambda a: _crop(store, a[0], short, crop, tuple(a[1]), bool(a[2])),
                                zip(b.image_ids, b.crop_yx, b.flip))
                x = to_model_input(np.stack(list(rows)))
                yield x, b.proxy_idx.astype(np.int32), np.int64(t)

    sig = (tf.TensorSpec((None, crop, crop, 3), tf.float32), tf.TensorSpec((None,), tf.int32),
           tf.TensorSpec((), tf.int64))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(prefetch)


def eval_arrays(image_ids, store, resize_short, crop, batch, degradation_id, seed, raw=False):
    """(ids, batch): preprocess_input'd, or float32 RGB in [0,255] when `raw`."""
    for s in range(0, len(image_ids), batch):
        ids = image_ids[s:s + batch]
        xs = []
        for i, iid in enumerate(ids, start=s):
            c = _crop(store, iid, resize_short, crop)
            if degradation_id != "none":
                rng = stream(seed, "degrade", i)
                c = degrade(c / 255.0, degradation_id, rng) * 255.0
            xs.append(c)
        x = np.stack(xs)
        yield ids, x.astype(np.float32) if raw else to_model_input(x)
