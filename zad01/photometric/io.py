"""Декодирование изображений в общее представление ФОТО-04: RGB float64 в [0, 1].

OpenCV не используется намеренно: порядок каналов всегда RGB (Pillow).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

SUPPORTED_PROFILES = ("sRGB", "convert_to_sRGB")


def prepare_image(path, color_profile: str = "sRGB") -> np.ndarray:
    """Читает файл и возвращает массив H×W×3 float64 RGB в [0, 1].

    color_profile:
      "sRGB"            — значения файла трактуются как закодированные sRGB;
                          встроенный ICC-профиль (если есть) не применяется;
      "convert_to_sRGB" — встроенный ICC-профиль (если есть) преобразуется в sRGB.
    """
    if color_profile not in SUPPORTED_PROFILES:
        raise ValueError(f"color_profile must be one of {SUPPORTED_PROFILES}, got {color_profile!r}")
    with Image.open(Path(path)) as im:
        im.load()
        if color_profile == "convert_to_sRGB" and im.info.get("icc_profile"):
            src = ImageCms.ImageCmsProfile(io.BytesIO(im.info["icc_profile"]))
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im.convert("RGB"), src, dst, outputMode="RGB")
        return pil_to_rgb01(im)


def pil_to_rgb01(im: Image.Image) -> np.ndarray:
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64)
        scale = 65535.0
        if arr.max(initial=0) > scale:
            raise ValueError(f"unsupported integer image range in mode {im.mode}")
        rgb = np.repeat((arr / scale)[..., None], 3, axis=2)
    else:
        if im.mode != "RGB":
            if "A" in im.mode or (im.mode == "P" and "transparency" in im.info):
                raise ValueError(f"images with alpha are not supported (mode {im.mode})")
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8).astype(np.float64) / 255.0
    return check_rgb01(rgb)


def check_rgb01(img: np.ndarray, name: str = "image") -> np.ndarray:
    """Проверка общего представления; защищает от смешения шкал [0,1] и [0,255]."""
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name}: expected H×W×3 RGB array, got shape {getattr(img, 'shape', None)}")
    if img.dtype != np.float64:
        raise ValueError(f"{name}: expected float64, got {img.dtype}")
    if img.size == 0:
        raise ValueError(f"{name}: empty image")
    if not np.isfinite(img).all():
        raise ValueError(f"{name}: non-finite values")
    lo, hi = float(img.min()), float(img.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"{name}: values must lie in [0, 1], got [{lo}, {hi}] (scale [0,255]?)")
    return img


def rgb01_to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def save_png(img: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb01_to_uint8(check_rgb01(img))).save(path, format="PNG")
