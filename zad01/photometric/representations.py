"""Общие преобразования представлений (ФОТО-04, Г1-24, таблица хешей)."""

from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dctn
from skimage.transform import resize

# Проектное яркостное представление: по закодированным значениям sRGB (не линейная яркость).
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

HIST_BINS_PER_CHANNEL = 8
HIST_BINS = HIST_BINS_PER_CHANNEL**3

DEFAULT_RESIZE = {"backend": "skimage", "order": 1, "anti_aliasing": False, "mode": "edge"}


def luminance(img: np.ndarray) -> np.ndarray:
    """Y = 0.2126 R + 0.7152 G + 0.0722 B, массив H×W float64."""
    return img[..., 0] * LUMA_WEIGHTS[0] + img[..., 1] * LUMA_WEIGHTS[1] + img[..., 2] * LUMA_WEIGHTS[2]


def resize_bilinear(y: np.ndarray, rows: int, cols: int, resize_cfg: dict | None = None) -> np.ndarray:
    """Единственная зафиксированная реализация bilinear resize для всех хешей."""
    cfg = {**DEFAULT_RESIZE, **(resize_cfg or {})}
    if cfg["backend"] != "skimage" or cfg["order"] != 1:
        raise ValueError(f"unsupported resize config {cfg}")
    return resize(
        y,
        (rows, cols),
        order=1,
        mode=cfg["mode"],
        anti_aliasing=bool(cfg["anti_aliasing"]),
        preserve_range=True,
        clip=False,
    ).astype(np.float64, copy=False)


def joint_rgb_histogram(img: np.ndarray) -> np.ndarray:
    """Г1-24: совместная RGB-гистограмма 8×8×8 с равными интервалами на [0, 1].

    Индекс канала = min(floor(v·8), 7): значение 1 относится к последнему интервалу.
    Возвращает 512 долей с суммой 1; каждый пиксель попадает ровно в один bin.
    """
    q = np.minimum(np.floor(img * HIST_BINS_PER_CHANNEL), HIST_BINS_PER_CHANNEL - 1).astype(np.int64)
    idx = (q[..., 0] * HIST_BINS_PER_CHANNEL + q[..., 1]) * HIST_BINS_PER_CHANNEL + q[..., 2]
    counts = np.bincount(idx.ravel(), minlength=HIST_BINS)
    return counts.astype(np.float64) / idx.size


def ahash_bits(y: np.ndarray, resize_cfg=None) -> np.ndarray:
    small = resize_bilinear(y, 8, 8, resize_cfg)
    return (small > small.mean()).ravel()


def dhash_bits(y: np.ndarray, resize_cfg=None) -> np.ndarray:
    # 9 столбцов × 8 строк: 8 горизонтальных разностей в каждой строке.
    small = resize_bilinear(y, 8, 9, resize_cfg)
    return (small[:, 1:] > small[:, :-1]).ravel()


def phash_bits(y: np.ndarray, resize_cfg=None) -> np.ndarray:
    # variant=dc_included: медиана всех 64 коэффициентов блока 8×8, включая DC.
    small = resize_bilinear(y, 32, 32, resize_cfg)
    block = dctn(small, type=2, norm="ortho")[:8, :8]
    return (block > np.median(block)).ravel()


def whash_bits(y: np.ndarray, resize_cfg=None) -> np.ndarray:
    # variant=haar_level2: Haar DWT, periodization, два уровня, блок cA2 8×8.
    small = resize_bilinear(y, 32, 32, resize_cfg)
    ca2 = pywt.wavedec2(small, "haar", mode="periodization", level=2)[0]
    if ca2.shape != (8, 8):
        raise AssertionError(f"unexpected cA2 shape {ca2.shape}")
    return (ca2 > np.median(ca2)).ravel()
