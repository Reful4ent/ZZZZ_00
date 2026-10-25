"""Input pipeline helpers; keep TF-free at module level."""


def resized_shape(h: int, w: int, short: int) -> tuple[int, int]:
    """(H', W') after aspect-preserving resize; long side is integer round-half-up."""
    s = min(h, w)
    def grow(x: int) -> int:
        return short if x == s else (x * short + s // 2) // s
    return grow(h), grow(w)


def center_offsets(h: int, w: int, crop: int) -> tuple[int, int]:
    return (h - crop) // 2, (w - crop) // 2


def resize_short(img, short: int):
    import cv2
    h, w = img.shape[:2]
    nh, nw = resized_shape(h, w, short)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def to_model_input(crop):
    """RGB uint8 or float in [0,255] -> caffe-style ResNet50 input; the only preprocess_input call."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import numpy as np
    return preprocess_input(np.array(crop, dtype=np.float32))


def degrade(crop01, degradation_id: str, rng):
    import numpy as np
    if degradation_id == "none":
        return crop01
    if degradation_id in ("bright_+0.1", "bright_-0.1"):
        return np.clip(crop01 + float(degradation_id[7:]), 0, 1)
    if degradation_id == "noise_0.03":
        return np.clip(crop01 + rng.normal(0, 0.03, crop01.shape), 0, 1)
    raise ValueError(f"unknown degradation {degradation_id!r}")
