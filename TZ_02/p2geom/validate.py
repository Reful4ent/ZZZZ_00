"""validate_geometry: q_geom (Г2-108…Г2-110) с перечнем причин отказа."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GeometryGateConfig
from .homography import Admissibility, check_admissible, hull_coverage
from .ransac import HomographyEstimate


def gate_failures(n_in, rho_in, med_err, c_I, c_E, cfg: GeometryGateConfig) -> dict[str, np.ndarray]:
    """Нарушения условий Г2-110 (кроме H_adm) — общая формула для одиночной
    проверки и для векторной калибровки. Аргументы — скаляры или массивы;
    med_err = NaN трактуется как нарушение."""
    out: dict[str, np.ndarray] = {}
    if cfg.use_support:
        med = np.nan_to_num(np.asarray(med_err, dtype=float), nan=np.inf)
        out["LOW_INLIERS"] = np.asarray(n_in) < cfg.n_min
        out["LOW_INLIER_RATIO"] = np.asarray(rho_in, dtype=float) < cfg.rho_min
        out["HIGH_MEDIAN_ERROR"] = med > cfg.eps_med
    if cfg.use_coverage:
        out["LOW_COVERAGE_I"] = np.asarray(c_I, dtype=float) < cfg.c_I_min
        out["LOW_COVERAGE_E"] = np.asarray(c_E, dtype=float) < cfg.c_E_min
    return out


@dataclass
class GeometryCheck:
    q_geom: int
    reasons: list[str] = field(default_factory=list)   # все нарушенные условия, в порядке проверки
    n0: int = 0
    n_in: int = 0
    rho_in: float | None = None
    med_err: float | None = None
    c_I: float = 0.0
    c_E: float = 0.0
    adm: Admissibility | None = None


def validate_geometry(est: HomographyEstimate, x: np.ndarray, u: np.ndarray,
                      size_I: tuple[int, int], size_E: tuple[int, int],
                      cfg: GeometryGateConfig) -> GeometryCheck:
    """x, u — точки M0 в исходных координатах I и E."""
    n0 = len(x)
    chk = GeometryCheck(q_geom=0, n0=n0)
    if n0 == 0:
        chk.reasons.append("NO_MATCHES")
        return chk
    if est.status != "OK" or est.H is None:
        chk.reasons.append(est.status if est.status != "OK" else "NO_MODEL")
        return chk
    inl = est.inliers
    chk.n_in = int(inl.sum())
    chk.rho_in = chk.n_in / n0
    chk.med_err = float(np.median(est.errors[inl])) if chk.n_in else None
    chk.c_I = hull_coverage(np.asarray(x)[inl], size_I)
    chk.c_E = hull_coverage(np.asarray(u)[inl], size_E)
    chk.adm = check_admissible(est.H, size_I, size_E, cfg.cond_max, cfg.w_min, cfg.area_ratio_range)
    fails = gate_failures(chk.n_in, chk.rho_in, np.nan if chk.med_err is None else chk.med_err,
                          chk.c_I, chk.c_E, cfg)
    chk.reasons.extend(name for name, bad in fails.items() if bool(bad))
    if not chk.adm.ok:
        chk.reasons.extend("H_" + r for r in chk.adm.reasons)
    chk.q_geom = int(not chk.reasons)
    return chk
