"""Управляемые воздействия (ФОТО-03, Ф-1) и контрпример с перестановкой пикселей (P1-E3).

Все воздействия строятся непосредственно из оригинала, а не поверх искажений KADID.
Результат хранится как PNG (8 бит) вместе с параметрами; оценивается именно
сохранённый и заново декодированный PNG. Данные помечаются synthetic и не являются
официальными уровнями KADID.
"""

from __future__ import annotations

import io as _io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from photometric.config import JPEG_SUBSAMPLING
from photometric.io import check_rgb01, pil_to_rgb01, prepare_image, rgb01_to_uint8, save_png


def photometric_transform(img: np.ndarray, a: float = 1.0, b: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """Ф-1: J = clip(a·(I − 0.5) + 0.5 + b, 0, 1)^γ.

    Порядок фиксирован: линейное преобразование → clip → γ.
    """
    if a == 1.0 and b == 0.0:
        lin = np.clip(img, 0.0, 1.0)  # контроль: точное тождество без ошибок округления
    else:
        lin = np.clip(a * (img - 0.5) + 0.5 + b, 0.0, 1.0)
    return lin if gamma == 1.0 else np.power(lin, gamma)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Независимый гауссов шум по каналам; clip в [0, 1] обязателен для хранения PNG."""
    noise = rng.normal(0.0, 1.0, size=img.shape) * sigma
    return np.clip(img + noise, 0.0, 1.0)


def jpeg_roundtrip(img: np.ndarray, quality: int, subsampling: str = "4:4:4", optimize: bool = False) -> np.ndarray:
    """Кодирование и декодирование через Pillow (libjpeg) с фиксированным chroma subsampling."""
    buf = _io.BytesIO()
    Image.fromarray(rgb01_to_uint8(img)).save(
        buf, format="JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING[subsampling], optimize=optimize
    )
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        return pil_to_rgb01(im)


def permute_pixels(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Перестановка пикселей целиком (RGB-тройки сохраняются): гистограмма неизменна."""
    h, w, c = img.shape
    flat = img.reshape(h * w, c)
    return flat[rng.permutation(h * w)].reshape(h, w, c)


def _original_number(original_id: str) -> int:
    return int(original_id.lstrip("I"))


def noise_rng(base_seed: int, original_id: str, sigma: float, k: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(base_seed), _original_number(original_id), int(round(sigma * 1e6)), int(k)])
    return np.random.Generator(np.random.PCG64(ss))


def _fmt(v) -> str:
    return f"{v:g}".replace("-", "m")


def e2_plan(original_ids, e2: dict) -> list[dict]:
    """Список синтетических запросов P1-E2: одиночные факторы при a=1, b=0, γ=1."""
    items = []
    for oid in sorted(original_ids):
        for b in e2["brightness"]:
            items.append(dict(original_id=oid, factor="brightness", param=float(b), a=1.0, b=float(b), gamma=1.0))
        for a in e2["contrast"]:
            items.append(dict(original_id=oid, factor="contrast", param=float(a), a=float(a), b=0.0, gamma=1.0))
        for g in e2["gamma"]:
            items.append(dict(original_id=oid, factor="gamma", param=float(g), a=1.0, b=0.0, gamma=float(g)))
        for s in e2["noise_sigma"]:
            for k in range(int(e2["noise_seeds"])):
                items.append(dict(original_id=oid, factor="noise", param=float(s), sigma=float(s), noise_seed=k))
        for q in e2["jpeg_quality"]:
            items.append(dict(original_id=oid, factor="jpeg", param=float(q), quality=int(q), **e2["jpeg"]))
    for it in items:
        suffix = f"_s{it['noise_seed']}" if it["factor"] == "noise" else ""
        it["query_id"] = f"S_{it['original_id']}_{it['factor']}_{_fmt(it['param'])}{suffix}"
    return items


def render_e2(item: dict, original: np.ndarray, e2: dict) -> np.ndarray:
    f = item["factor"]
    if f in ("brightness", "contrast", "gamma"):
        return photometric_transform(original, a=item["a"], b=item["b"], gamma=item["gamma"])
    if f == "noise":
        rng = noise_rng(e2["noise_seed"], item["original_id"], item["sigma"], item["noise_seed"])
        return gaussian_noise(original, item["sigma"], rng)
    if f == "jpeg":
        return jpeg_roundtrip(original, item["quality"], item["subsampling"], item["optimize"])
    raise ValueError(f"unknown factor {f}")


def materialize(items: list[dict], original_paths: dict[str, str], out_dir: Path, render, color_profile="sRGB") -> pd.DataFrame:
    """Генерирует PNG и params.json; уже сгенерированные файлы с теми же параметрами переиспользуются."""
    out_dir = Path(out_dir)
    rows = []
    cache: dict[str, np.ndarray] = {}
    for it in items:
        oid = it["original_id"]
        png = out_dir / oid / f"{it['query_id']}.png"
        meta_path = png.with_suffix(".json")
        params = {**it, "synthetic": True, "source_original": Path(original_paths[oid]).name, "formula": "F-1" if it["factor"] in ("brightness", "contrast", "gamma") else it["factor"]}
        if not (png.is_file() and meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) == params):
            if oid not in cache:
                cache.clear()
                cache[oid] = prepare_image(original_paths[oid], color_profile)
            img = check_rgb01(render(it, cache[oid]))
            save_png(img, png)
            meta_path.write_text(json.dumps(params, indent=1, sort_keys=True), encoding="utf-8")
        rows.append({**it, "query_path": str(png), "synthetic": True})
    return pd.DataFrame(rows)


def libjpeg_version() -> str | None:
    from PIL import features

    return features.version("jpg")
