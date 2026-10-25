"""compare_regions: участки, доступные площади, фотометрическая компенсация,
d_k и a_k (Г2-112…Г2-118, Г2-120).

Сетка фиксирована в координатах E; при неделимости размеров последние
строка/столбец получают остаток пикселей. Участки вне K_val получают
d_k = a_k = NaN (null), а не 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from .color import delta_e00, srgb_to_lab
from .config import CompareConfig


def grid_edges(n_pixels: int, n: int) -> np.ndarray:
    if n_pixels < n:
        raise ValueError(f"{n_pixels} px cannot be split into {n} non-empty cells")
    s = n_pixels // n
    return np.array([i * s for i in range(n)] + [n_pixels])


def cell_index_map(size: tuple[int, int], n: int) -> np.ndarray:
    W, H = size
    re, ce = grid_edges(H, n), grid_edges(W, n)
    r = np.searchsorted(re, np.arange(H), side="right") - 1
    c = np.searchsorted(ce, np.arange(W), side="right") - 1
    return r[:, None] * n + c[None, :]


def critical_cells(n: int, mode: str) -> list[int]:
    """K_req = K_crit. center4: четыре центральных участка (задаются до генерации дефекта)."""
    if mode == "none":
        return []
    if mode == "center4":
        if n % 2:
            raise ValueError("center4 requires an even grid")
        m = n // 2
        return [r * n + c for r in (m - 1, m) for c in (m - 1, m)]
    raise ValueError(mode)


def calibration_band(size: tuple[int, int], band: float) -> np.ndarray:
    """Фиксированная опорная полоса шириной band по периметру E."""
    W, H = size
    bw, bh = int(round(band * W)), int(round(band * H))
    m = np.zeros((H, W), dtype=bool)
    m[:bh, :] = m[H - bh:, :] = True
    m[:, :bw] = m[:, W - bw:] = True
    return m


@dataclass
class RegionMetrics:
    grid: int
    tau_pix: tuple[float, ...]
    area_full: np.ndarray        # |R_k|
    area_avail: np.ndarray       # |Ω_k|
    v: np.ndarray                # v_k
    kval: np.ndarray             # bool, k ∈ K_val
    d_mean: np.ndarray           # d_k (Г2-118), NaN вне K_val
    d_of_means: np.ndarray       # ΔE00 средних Lab участка (P2-E5), NaN вне K_val
    a: np.ndarray                # (len(tau_pix), K) доли превышений, NaN вне K_val
    w: np.ndarray                # нормированные веса, 0 вне K_val
    critical: list[int]
    compensation: dict = field(default_factory=dict)
    delta_map: np.ndarray | None = None   # δ00 на Ω_cmp, NaN вне

    def d(self, stat: str) -> np.ndarray:
        if stat == "mean_delta":
            return self.d_mean
        if stat == "delta_of_means":
            return self.d_of_means
        raise ValueError(stat)

    def a_at(self, tau_pix: float) -> np.ndarray:
        idx = [i for i, t in enumerate(self.tau_pix) if t == tau_pix]
        if not idx:
            raise ValueError(f"tau_pix={tau_pix} not computed (have {self.tau_pix})")
        return self.a[idx[0]]


def photometric_compensation(J: np.ndarray, E: np.ndarray, cal: np.ndarray,
                             a_bounds, b_bounds) -> tuple[float, float, str]:
    """Один общий (a, b) на три канала RGB: min Σ (a I + b − E)² на опорной полосе
    при a∈a_bounds, b∈b_bounds (ограниченный линейный МНК; clip применяется после)."""
    if not cal.any():
        return 1.0, 0.0, "CAL_EMPTY"
    x = J[cal].reshape(-1)
    y = E[cal].reshape(-1)
    A = np.column_stack([x, np.ones_like(x)])
    res = lsq_linear(A, y, bounds=([a_bounds[0], b_bounds[0]], [a_bounds[1], b_bounds[1]]))
    return float(res.x[0]), float(res.x[1]), "OK" if res.success else "LSQ_FAILED"


def compare_regions(E_rgb: np.ndarray, I_al: np.ndarray, omega_cmp: np.ndarray,
                    cfg: CompareConfig, tau_pix: tuple[float, ...] | None = None,
                    keep_map: bool = True) -> RegionMetrics:
    """E_rgb, I_al — float [0,1] в исходном разрешении E; omega_cmp = Ω̂_com ∩ Ω_obj."""
    H, W = E_rgb.shape[:2]
    n = cfg.grid
    taus = tuple(sorted(set(tau_pix if tau_pix is not None else cfg.tau_pix_grid)))
    cells = cell_index_map((W, H), n)
    K = n * n
    area_full = np.bincount(cells.ravel(), minlength=K)
    comp = {"enabled": cfg.compensation, "a": 1.0, "b": 0.0, "status": "OFF", "n_cal": 0}
    J = I_al
    if cfg.compensation:
        cal = calibration_band((W, H), cfg.cal_band) & omega_cmp
        a_, b_, st = photometric_compensation(I_al, E_rgb, cal, cfg.a_bounds, cfg.b_bounds)
        comp.update(a=a_, b=b_, status=st, n_cal=int(cal.sum()))
        J = np.clip(a_ * I_al + b_, 0.0, 1.0)
    idx = cells[omega_cmp]
    if idx.size:
        lab_J = srgb_to_lab(J[omega_cmp])
        lab_E = srgb_to_lab(E_rgb[omega_cmp])
        delta = delta_e00(lab_J, lab_E)
    else:
        lab_J = lab_E = np.zeros((0, 3))
        delta = np.zeros(0)
    area = np.bincount(idx, minlength=K)
    v = area / area_full
    kval = (v >= cfg.v_min) & (area >= cfg.n_min_area)
    safe = np.where(area > 0, area, 1)
    d_mean = np.where(kval, np.bincount(idx, weights=delta, minlength=K) / safe, np.nan)
    a = np.stack([np.where(kval, np.bincount(idx, weights=(delta > t).astype(float), minlength=K) / safe, np.nan)
                  for t in taus])
    mJ = np.stack([np.bincount(idx, weights=lab_J[:, c], minlength=K) for c in range(3)], 1) / safe[:, None]
    mE = np.stack([np.bincount(idx, weights=lab_E[:, c], minlength=K) for c in range(3)], 1) / safe[:, None]
    d_of_means = np.where(kval, delta_e00(mJ, mE), np.nan)
    tot = area[kval].sum()
    w = np.where(kval, area / tot, 0.0) if tot > 0 else np.zeros(K)
    dmap = None
    if keep_map:
        dmap = np.full((H, W), np.nan)
        dmap[omega_cmp] = delta
    return RegionMetrics(n, taus, area_full, area, v, kval, d_mean, d_of_means, a, w,
                         critical_cells(n, cfg.critical), comp, dmap)
