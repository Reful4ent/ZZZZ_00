"""warp_with_mask (Г2-99) и доступная область / q_align (Г2-100, Г2-111).

Для каждого пикселя u целевого изображения исходное изображение
выбирается в точке π(H⁻¹u), где H: исходное -> целевое. Цвет — bilinear,
категориальные маски — nearest. Пиксель целевого изображения наблюдаем,
только если все пиксели опорной окрестности существуют (и видимы, если
задана маска видимости источника). Заливка вне области — 0 и не считается
наблюдаемой.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AlignConfig
from .homography import image_corners, to_homogeneous


def _orient_positive(H: np.ndarray, src_size: tuple[int, int]) -> np.ndarray:
    """Знак H выбирается так, чтобы w > 0 в углах исходного изображения."""
    w = to_homogeneous(image_corners(src_size)) @ H[2]
    return -H if np.all(w < 0) else H


def source_coords(H: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int],
                  w_eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Координаты выборки (xs, ys) для каждого пикселя цели и маска w > w_eps."""
    H = _orient_positive(np.asarray(H, dtype=np.float64), src_size)
    Hinv = np.linalg.inv(H)
    Wd, Hd = dst_size
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float64)
    X = Hinv[0, 0] * xx + Hinv[0, 1] * yy + Hinv[0, 2]
    Y = Hinv[1, 0] * xx + Hinv[1, 1] * yy + Hinv[1, 2]
    Wh = Hinv[2, 0] * xx + Hinv[2, 1] * yy + Hinv[2, 2]
    ok = Wh > w_eps
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = np.where(ok, X / Wh, np.nan)
        ys = np.where(ok, Y / Wh, np.nan)
    return xs, ys, ok


def warp_with_mask(src: np.ndarray, H: np.ndarray, dst_size: tuple[int, int],
                   src_visible: np.ndarray | None = None, interp: str = "bilinear",
                   w_eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (изображение в координатах цели, маска наблюдаемости Ω̂_com)."""
    src = np.asarray(src)
    Hs, Ws = src.shape[:2]
    xs, ys, ok = source_coords(H, (Ws, Hs), dst_size, w_eps)
    Wd, Hd = dst_size
    out_shape = (Hd, Wd) + src.shape[2:]
    if interp == "nearest":
        xi = np.floor(np.nan_to_num(xs, nan=-1.0) + 0.5).astype(np.int64)
        yi = np.floor(np.nan_to_num(ys, nan=-1.0) + 0.5).astype(np.int64)
        valid = ok & (xi >= 0) & (xi < Ws) & (yi >= 0) & (yi < Hs)
        xi, yi = np.where(valid, xi, 0), np.where(valid, yi, 0)
        if src_visible is not None:
            valid &= src_visible[yi, xi].astype(bool)
        out = np.zeros(out_shape, dtype=src.dtype)
        out[valid] = src[yi[valid], xi[valid]]
        return out, valid
    if interp != "bilinear":
        raise ValueError(interp)
    if Ws < 2 or Hs < 2:
        raise ValueError("bilinear warp needs source >= 2x2")
    valid = ok & (xs >= 0) & (xs <= Ws - 1) & (ys >= 0) & (ys <= Hs - 1)
    xs = np.where(valid, xs, 0.0)
    ys = np.where(valid, ys, 0.0)
    x0 = np.minimum(np.floor(xs).astype(np.int64), Ws - 2)
    y0 = np.minimum(np.floor(ys).astype(np.int64), Hs - 2)
    fx, fy = xs - x0, ys - y0
    if src_visible is not None:
        vis = src_visible.astype(bool)
        valid &= vis[y0, x0] & vis[y0, x0 + 1] & vis[y0 + 1, x0] & vis[y0 + 1, x0 + 1]
    s = src.astype(np.float64)
    if s.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    out = ((1 - fy) * ((1 - fx) * s[y0, x0] + fx * s[y0, x0 + 1])
           + fy * ((1 - fx) * s[y0 + 1, x0] + fx * s[y0 + 1, x0 + 1]))
    out[~valid] = 0.0
    return out, valid


def object_mask(size: tuple[int, int], border_px: int = 0) -> np.ndarray:
    """Ω_obj по умолчанию: изображение E без исключаемых границ (не маска дефекта)."""
    W, H = size
    m = np.zeros((H, W), dtype=bool)
    b = int(border_px)
    m[b:H - b if b else H, b:W - b if b else W] = True
    return m


@dataclass
class AlignQuality:
    q_align: int
    rho_vis: float | None
    area: int
    reasons: list[str] = field(default_factory=list)


def align_failures(rho_vis, area, cfg: AlignConfig) -> dict[str, np.ndarray]:
    """Нарушения Г2-111 (кроме q_geom); скаляры или массивы, NaN — нарушение."""
    rho = np.nan_to_num(np.asarray(rho_vis, dtype=float), nan=-np.inf)
    return {"LOW_VISIBILITY": rho < cfg.rho_vis_min, "SMALL_COMMON_AREA": np.asarray(area) < cfg.a_min}


def alignment_quality(valid: np.ndarray, omega_obj: np.ndarray, q_geom: int,
                      cfg: AlignConfig) -> AlignQuality:
    obj = int(omega_obj.sum())
    if obj == 0:
        return AlignQuality(0, None, 0, ["EMPTY_OBJECT_REGION"])
    area = int((valid & omega_obj).sum())
    rho = area / obj
    reasons = [] if q_geom else ["Q_GEOM_0"]
    reasons += [name for name, bad in align_failures(rho, area, cfg).items() if bool(bad)]
    return AlignQuality(int(not reasons), rho, area, reasons)
