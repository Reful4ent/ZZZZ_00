"""sRGB -> CIELAB (D65, 2°) и CIEDE2000 (kL = kC = kH = 1), Г2-117."""
from __future__ import annotations

import numpy as np
from skimage import color as skcolor


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: float в [0, 1], последняя ось — каналы (любая форма (..., 3))."""
    rgb = np.asarray(rgb, dtype=np.float64)
    flat = rgb.reshape(-1, 1, 3)
    return skcolor.rgb2lab(flat, illuminant="D65", observer="2").reshape(rgb.shape)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    flat = lab.reshape(-1, 1, 3)
    return np.clip(skcolor.lab2rgb(flat, illuminant="D65", observer="2").reshape(lab.shape), 0.0, 1.0)


def delta_e00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return skcolor.deltaE_ciede2000(np.asarray(lab1, np.float64), np.asarray(lab2, np.float64),
                                    kL=1, kC=1, kH=1, channel_axis=-1)
