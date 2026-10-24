"""Seeds and named random streams (spec §7.2, ВОСП-02)."""

import random
import sys

import cv2
import numpy as np

STREAM_INDEX = {"sampler": 0, "augment": 1, "init": 2, "bootstrap": 3, "degrade": 4}

_last_seed: int | None = None


def init_seeds(seed: int) -> None:
    """Seed global generators; TensorFlow only if the caller already imported it (it is loaded lazily)."""
    global _last_seed
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
    _last_seed = seed


def stream(seed: int, name: str, unit_id: int) -> np.random.Generator:
    """Independent generator bound to (seed, stream name, unit of work): resume-safe by construction."""
    if name not in STREAM_INDEX:
        raise ValueError(f"unknown stream {name!r}; known {sorted(STREAM_INDEX)}")
    seq = np.random.SeedSequence(seed, spawn_key=(STREAM_INDEX[name], unit_id))
    return np.random.Generator(np.random.PCG64(seq))


def capture_rng_state() -> dict:
    """JSON-serialisable snapshot of global RNG states; OpenCV's state cannot be read, only its seed."""
    version, internal, gauss = random.getstate()
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    tf = sys.modules.get("tensorflow")
    tf_state = {"available": False}
    if tf is not None:
        tf_state = {"available": True, "seed": _last_seed}
        try:
            tf_state["global_generator_state"] = tf.random.get_global_generator().state.numpy().tolist()
        except Exception as e:   # not all TF versions expose it
            tf_state["global_generator_state"] = None
            tf_state["note"] = f"global_generator_state_unavailable: {type(e).__name__}"
    return {
        "seed": _last_seed,
        "python_random": {"version": version, "state": list(internal), "gauss": gauss},
        "numpy_global": {"kind": kind, "keys": keys.tolist(), "pos": int(pos), "has_gauss": int(has_gauss),
                         "cached_gaussian": float(cached)},
        "streams": {"generator": "PCG64(SeedSequence(seed, spawn_key=(stream_index, unit_id)))",
                    "stream_index": dict(STREAM_INDEX)},
        "tensorflow": tf_state,
        "opencv": {"seed": _last_seed, "state": None, "note": "opencv_state_not_readable"},
    }
