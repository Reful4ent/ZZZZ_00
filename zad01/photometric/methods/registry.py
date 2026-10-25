"""Реестр зафиксированных методов P1.

Каждый method_id — один однозначный вариант. Изменение варианта (например, хеш
другой публикации или сравнение средних цветов) регистрируется под новым method_id.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from photometric.methods import pixel
from photometric.methods.base import DISTANCE, SIMILARITY, UNDEFINED_EPS, SIZE_MISMATCH, Result
from photometric import representations as R

HASH_BITS = 64


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    kind: str  # DISTANCE | SIMILARITY
    represent_fn: Callable[[np.ndarray, dict], np.ndarray]
    compare_fn: Callable[[np.ndarray, np.ndarray, dict], Result]
    defaults: dict = field(default_factory=dict)
    description: str = ""
    # Представление содержит попиксельные данные (большое) или компактный вектор.
    pixel_level: bool = False


def _hamming(a: np.ndarray, b: np.ndarray, params: dict) -> Result:
    """Ф-4: нормированная дистанция Хэмминга, m = 64."""
    if a.shape != (HASH_BITS,) or b.shape != (HASH_BITS,):
        raise ValueError("hash representations must have 64 bits")
    return Result(float(np.count_nonzero(a != b)) / HASH_BITS)


def _hist_intersection(a: np.ndarray, b: np.ndarray, params: dict) -> Result:
    """Г1-25: Σ_k min(h_I(k), h_E(k)); больше — большее совпадение."""
    return Result(float(np.minimum(a, b).sum()))


def _ciede_mean(a: np.ndarray, b: np.ndarray, params: dict) -> Result:
    """Г1-27: среднее попиксельных ΔE00 (Г1-26), эталонный backend skimage."""
    if a.shape != b.shape:
        return Result.undefined(SIZE_MISMATCH)
    de = deltaE_ciede2000(a, b, kL=params["kL"], kC=params["kC"], kH=params["kH"], channel_axis=-1)
    return Result(float(np.mean(de)))


def _lab(img: np.ndarray, params: dict) -> np.ndarray:
    return rgb2lab(img, illuminant=params["illuminant"], observer=params["observer"], channel_axis=-1)


def _hash_spec(method_id: str, fn, variant: str, description: str) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        kind=DISTANCE,
        represent_fn=lambda img, p: fn(R.luminance(img), p["resize"]),
        compare_fn=_hamming,
        defaults={"variant": variant, "bits": HASH_BITS, "resize": dict(R.DEFAULT_RESIZE), "luma": "bt709_encoded_srgb"},
        description=description,
    )


METHODS: dict[str, MethodSpec] = {
    spec.method_id: spec
    for spec in [
        MethodSpec(
            "mse",
            DISTANCE,
            lambda img, p: img,
            lambda a, b, p: pixel.mse(a, b),
            {"space": "rgb01", "channels": 3},
            "Г1-20: MSE по пикселям и каналам RGB [0,1]",
            pixel_level=True,
        ),
        MethodSpec(
            "ncc",
            SIMILARITY,
            lambda img, p: R.luminance(img),
            lambda a, b, p: pixel.ncc(a, b, p["eps"]),
            {"space": "Y_bt709_encoded_srgb", "eps": UNDEFINED_EPS},
            "Ф-2: NCC по яркости Y",
            pixel_level=True,
        ),
        MethodSpec(
            "zncc",
            SIMILARITY,
            lambda img, p: R.luminance(img),
            lambda a, b, p: pixel.zncc(a, b, p["eps"]),
            {"space": "Y_bt709_encoded_srgb", "eps": UNDEFINED_EPS},
            "Ф-3: ZNCC по яркости Y",
            pixel_level=True,
        ),
        MethodSpec(
            "hist_rgb8",
            SIMILARITY,
            lambda img, p: R.joint_rgb_histogram(img),
            _hist_intersection,
            {"bins": [8, 8, 8], "range": [0.0, 1.0], "last_bin_closed": True},
            "Г1-24/Г1-25: пересечение совместных RGB-гистограмм 8×8×8",
        ),
        MethodSpec(
            "ciede2000_mean",
            DISTANCE,
            _lab,
            _ciede_mean,
            {"illuminant": "D65", "observer": "2", "kL": 1, "kC": 1, "kH": 1, "backend": "skimage.color.deltaE_ciede2000"},
            "Г1-26/Г1-27: среднее попиксельное CIEDE2000",
            pixel_level=True,
        ),
        _hash_spec("ahash64", R.ahash_bits, "mean_strict_rowmajor", "aHash64: 8×8, бит = Y > mean"),
        _hash_spec("dhash64", R.dhash_bits, "horizontal_9x8", "dHash64: 9×8, бит = правый > левый"),
        _hash_spec("phash64", R.phash_bits, "dc_included", "pHash64: 32×32, DCT-II ortho, 8×8, медиана с DC"),
        _hash_spec("whash64", R.whash_bits, "haar_level2", "wHash64: 32×32, Haar periodization L2, cA2 8×8"),
    ]
}

DEFAULT_METHOD_IDS = list(METHODS)


_RESOLVED: dict[str, tuple] = {}


def resolve(method_config) -> tuple[MethodSpec, dict, str]:
    """method_config: method_id (str) или dict {"method_id": ..., **overrides}."""
    key = json.dumps(method_config, sort_keys=True, default=str)
    if key not in _RESOLVED:
        _RESOLVED[key] = _resolve(method_config)
    spec, params, chash = _RESOLVED[key]
    return spec, copy.deepcopy(params), chash


def _resolve(method_config) -> tuple[MethodSpec, dict, str]:
    if isinstance(method_config, str):
        method_config = {"method_id": method_config}
    method_id = method_config["method_id"]
    if method_id not in METHODS:
        raise KeyError(f"unknown method_id {method_id!r}; known: {list(METHODS)}")
    spec = METHODS[method_id]
    overrides = {k: v for k, v in method_config.items() if k != "method_id"}
    unknown = set(overrides) - set(spec.defaults)
    if unknown:
        raise KeyError(f"{method_id}: unknown parameters {sorted(unknown)}")
    params = {**spec.defaults, **overrides}
    if "resize" in overrides:
        params["resize"] = {**R.DEFAULT_RESIZE, **overrides["resize"]}
    return spec, params, config_hash(method_id, params)


def config_hash(method_id: str, params: dict) -> str:
    blob = json.dumps({"method_id": method_id, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
