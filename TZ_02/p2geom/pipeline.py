"""Конвейер одной пары: признаки -> пары -> RANSAC -> проверки -> совмещение ->
общая область -> участки -> решение с отказом.

GT (истинная H, маски дефектов) конвейеру не передаётся. Режим alignment="oracle"
(P2-E3) получает H_gt явно и помечается как контрольная ветвь, не рабочий метод.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .decision import Decision, decide
from .features import Features, detect_describe
from .homography import check_admissible, resize_crop_matrix
from .matching import Matches, match_descriptors
from .ransac import HomographyEstimate, estimate_homography
from .regions import RegionMetrics, compare_regions
from .validate import GeometryCheck, validate_geometry
from .warp import AlignQuality, alignment_quality, object_mask, warp_with_mask


def image_size(img: np.ndarray) -> tuple[int, int]:
    return img.shape[1], img.shape[0]


@dataclass
class PairResult:
    pair_id: str
    alignment: str                         # estimated | oracle | none
    method: str
    fI: Features | None = None
    fE: Features | None = None
    matches: Matches | None = None
    est: HomographyEstimate | None = None
    geom: GeometryCheck | None = None
    H: np.ndarray | None = None            # I -> E, использованная для совмещения
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # M0, точки I (исх.)
    u: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # M0, точки E (исх.)
    aligned: np.ndarray | None = None
    valid: np.ndarray | None = None
    align: AlignQuality | None = None
    regions: RegionMetrics | None = None
    decision: Decision | None = None
    timings: dict = field(default_factory=dict)

    @property
    def q_geom(self) -> int:
        return self.geom.q_geom if self.geom else 0

    @property
    def q_align(self) -> int:
        return self.align.q_align if self.align else 0

    @property
    def q_cmp(self) -> int:
        return self.decision.q_cmp if self.decision else 0

    @property
    def y_pred(self) -> int | None:
        return self.decision.y_pred if self.decision else None

    def reject_reason(self) -> str:
        """Первая причина отказа по порядку стадий ('' — отказа нет)."""
        for stage in (self.geom.reasons if self.geom else ["NOT_RUN"],
                      [r for r in (self.align.reasons if self.align else []) if r != "Q_GEOM_0"],
                      [r for r in (self.decision.reasons if self.decision else []) if r != "Q_ALIGN_0"]):
            if stage:
                return stage[0]
        return ""


def run_geometry(E: np.ndarray, I: np.ndarray, cfg: Config, pair_id: str = "",
                 fI: Features | None = None, fE: Features | None = None) -> PairResult:
    """Ветвь geometry (HPatches): признаки, M0, Ĥ, q_geom. fI/fE — готовые признаки
    (кэш), должны быть получены с тем же cfg.features."""
    res = PairResult(pair_id, "estimated", cfg.features.method)
    size_I, size_E = image_size(I), image_size(E)
    res.fI = fI if fI is not None else detect_describe(I, cfg.features)
    res.fE = fE if fE is not None else detect_describe(E, cfg.features)
    res.timings["detect"] = res.fI.seconds + res.fE.seconds
    if res.fI.status != "OK" or res.fE.status != "OK":
        res.geom = GeometryCheck(0, ["FEATURES_NOT_RUN"])
        return res
    res.matches = match_descriptors(res.fI, res.fE, cfg.matching.ratio)
    res.timings["match"] = res.matches.seconds
    res.x = res.fI.kp_orig[res.matches.pairs[:, 0]]
    res.u = res.fE.kp_orig[res.matches.pairs[:, 1]]
    g = cfg.gate

    def admissible(H):
        return check_admissible(H, size_I, size_E, g.cond_max, g.w_min, g.area_ratio_range).ok

    res.est = estimate_homography(res.x, res.u, cfg.ransac, admissible)
    res.timings["ransac"] = res.est.seconds
    res.geom = validate_geometry(res.est, res.x, res.u, size_I, size_E, g)
    res.H = res.est.H
    return res


def _forced_geometry(H: np.ndarray, size_I, size_E, cfg: Config) -> GeometryCheck:
    adm = check_admissible(H, size_I, size_E, cfg.gate.cond_max, cfg.gate.w_min)
    return GeometryCheck(int(adm.ok), [] if adm.ok else ["H_" + r for r in adm.reasons], adm=adm)


def run_pair(E: np.ndarray, I: np.ndarray, cfg: Config, pair_id: str = "", *,
             vis_mask_I: np.ndarray | None = None, alignment: str = "estimated",
             H_oracle: np.ndarray | None = None, keep_map: bool = True,
             fI: Features | None = None, fE: Features | None = None) -> PairResult:
    """Полный конвейер inspection. E, I — RGB float [0,1].
    vis_mask_I — маска видимости I (только режим known-mask, ГЕОМ-04)."""
    size_I, size_E = image_size(I), image_size(E)
    if alignment == "estimated":
        res = run_geometry(E, I, cfg, pair_id, fI, fE)
    elif alignment == "oracle":
        if H_oracle is None:
            raise ValueError("oracle alignment needs H_oracle")
        res = PairResult(pair_id, "oracle", "oracle", H=np.asarray(H_oracle, float))
        res.geom = _forced_geometry(res.H, size_I, size_E, cfg)
    elif alignment == "none":
        # Без совмещения: только приведение размеров I к E.
        res = PairResult(pair_id, "none", "none", H=resize_crop_matrix(size_I, size_E))
        res.geom = _forced_geometry(res.H, size_I, size_E, cfg)
    else:
        raise ValueError(alignment)
    t0 = time.perf_counter()
    omega_obj = object_mask(size_E, cfg.align.border_px)
    usable = (res.H is not None and np.all(np.isfinite(res.H))
              and abs(np.linalg.det(res.H)) > 0)
    if not usable:
        res.align = AlignQuality(0, None, 0, ["NO_MODEL"])
        res.decision = decide(0, None, cfg.decision, cfg.compare)
        return res
    res.aligned, res.valid = warp_with_mask(I, res.H, size_E, vis_mask_I, "bilinear", cfg.ransac.w_eps)
    res.align = alignment_quality(res.valid, omega_obj, res.q_geom, cfg.align)
    res.timings["warp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    taus = tuple(sorted(set(cfg.compare.tau_pix_grid) | {cfg.decision.tau_pix}))
    res.regions = compare_regions(E, res.aligned, res.valid & omega_obj, cfg.compare, taus, keep_map)
    res.decision = decide(res.q_align, res.regions, cfg.decision, cfg.compare)
    res.timings["compare"] = time.perf_counter() - t0
    return res
