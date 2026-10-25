"""Контролируемые воздействия (раздел 2) и синтетические изменения объекта (ОБЛ-01, ОБЛ-02).

Порядок операций (Г-2): изменение объекта в координатах E -> геометрия E -> I ->
фотометрия -> шум -> окклюзия -> PNG. Генератор хранит матрицу G: E -> I;
рабочее направление H_gt = G⁻¹: I -> E. Матрица генератора — GT, конвейеру
она не передаётся.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import delta_e00, lab_to_srgb, srgb_to_lab
from .homography import dlt, image_corners, project, quad_orientation_signs
from .warp import warp_with_mask

# ---------------------------------------------------------------- геометрия

FACTOR_GRID: dict[str, list[float]] = {
    "rotation": [0.0, 15.0, -15.0, 30.0, -30.0, 60.0, -60.0],
    "scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "shear": [-0.2, -0.1, 0.0, 0.1, 0.2],
    "perspective": [0.0, 0.03, 0.06, 0.10],
    "occlusion": [0.0, 0.10, 0.25, 0.40],
}
PERSPECTIVE_SEEDS = 5
OCCLUDER_ASPECTS = (0.5, 1.0, 2.0)
# Допускаемый диапазон отношения площадей перенесённого контура I -> E для H_adm
# в синтетике: масштаб 0.5…1.5 даёт 1/s² = 0.44…4.0; запас на перспективу.
SYNTH_AREA_RATIO_RANGE = (0.25, 6.0)


def _about_center(M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    W, H = size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    return T @ M @ np.linalg.inv(T)


def rotation_matrix(deg: float, size) -> np.ndarray:
    t = np.deg2rad(deg)
    return _about_center(np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]]), size)


def scale_matrix(s: float, size) -> np.ndarray:
    return _about_center(np.diag([s, s, 1.0]), size)


def shear_matrix(k: float, size) -> np.ndarray:
    return _about_center(np.array([[1, k, 0], [0, 1, 0], [0, 0, 1.0]]), size)


def perspective_matrix(r: float, size, rng: np.random.Generator,
                       max_attempts: int = 1000) -> tuple[np.ndarray, int]:
    """Каждая координата каждого угла смещается равномерно в [−rD, rD], D — диагональ.
    Повторная генерация при самопересечении или смене ориентации."""
    W, H = size
    D = float(np.hypot(W, H))
    src = image_corners(size)
    ref = quad_orientation_signs(src)
    for attempt in range(1, max_attempts + 1):
        dst = src + rng.uniform(-r * D, r * D, size=(4, 2))
        if np.all(quad_orientation_signs(dst) == ref):
            G = dlt(src, dst)
            if G is not None:
                return G / G[2, 2], attempt
    raise RuntimeError(f"perspective r={r}: no valid quad in {max_attempts} attempts")


# ---------------------------------------------------------------- окклюзия / рендер


def occluder(size, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Один прямоугольник площади frac (отношение сторон из {0.5, 1, 2}),
    случайный центр; прямоугольник целиком внутри кадра. True — закрыто."""
    W, H = size
    m = np.zeros((H, W), dtype=bool)
    if frac <= 0:
        return m, {"occ_rect": None, "occ_area_px": 0, "occ_frac": 0.0, "occ_aspect": None}
    aspect = float(rng.choice(OCCLUDER_ASPECTS))
    area = frac * W * H
    w = int(round(min(W, np.sqrt(area * aspect))))
    h = int(round(min(H, area / max(w, 1))))
    x0 = int(rng.integers(0, W - w + 1))
    y0 = int(rng.integers(0, H - h + 1))
    m[y0:y0 + h, x0:x0 + w] = True
    a = int(m.sum())
    return m, {"occ_rect": [x0, y0, w, h], "occ_area_px": a, "occ_frac": a / (W * H), "occ_aspect": aspect}


def png_roundtrip(img: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def observed_in_E(G: np.ndarray, size_E, size_I, occ_I: np.ndarray) -> np.ndarray:
    """Маска пикселей E, наблюдаемых в I: G(u) внутри I (опора bilinear) и не закрыто."""
    W, H = size_E
    Wi, Hi = size_I
    yy, xx = np.mgrid[0:H, 0:W]
    p, ok = project(G, np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64))
    px = np.nan_to_num(p[:, 0], nan=-1.0)
    py = np.nan_to_num(p[:, 1], nan=-1.0)
    inside = ok & (px >= 0) & (px <= Wi - 1) & (py >= 0) & (py <= Hi - 1)
    xi = np.clip(np.floor(px + 0.5).astype(int), 0, Wi - 1)
    yi = np.clip(np.floor(py + 0.5).astype(int), 0, Hi - 1)
    inside &= ~occ_I[yi, xi]
    return inside.reshape(H, W)


@dataclass
class Observation:
    I: np.ndarray                 # наблюдение после PNG
    G: np.ndarray                 # E -> I (генератор)
    H_gt: np.ndarray              # I -> E
    visible_I: np.ndarray         # маска видимости I (known-mask режим)
    occ_I: np.ndarray
    visibility: float             # доля пикселей E, наблюдаемых в I
    defect_visible_frac: float | None
    meta: dict = field(default_factory=dict)


def observe(E: np.ndarray, G: np.ndarray, rng: np.random.Generator, *, brightness: float = 0.0,
            noise_sigma: float = 0.0, occlusion: float = 0.0, defect_mask_E: np.ndarray | None = None,
            min_defect_visible: float | None = None, max_resample: int = 200,
            occ_I: np.ndarray | None = None, occ_meta: dict | None = None) -> Observation:
    """T(E): геометрия -> фотометрия -> шум -> окклюзия -> PNG.
    Если задан min_defect_visible, окклюдер пересэмплируется, пока видимая доля
    дефекта не станет >= порога (число пересэмплирований — в meta).
    occ_I/occ_meta позволяют повторить тот же окклюдер для парной ветви."""
    H, W = E.shape[:2]
    size = (W, H)
    I, from_E = warp_with_mask(E, G, size)
    I = np.clip(I + brightness, 0.0, 1.0)
    if noise_sigma > 0:
        I = np.clip(I + rng.normal(0.0, noise_sigma, I.shape), 0.0, 1.0)
    resamples = 0
    if occ_I is None:
        while True:
            occ_I, occ_meta = occluder(size, occlusion, rng)
            obs = observed_in_E(G, size, size, occ_I)
            dvis = None
            if defect_mask_E is not None and defect_mask_E.any():
                dvis = float(obs[defect_mask_E].mean())
            if min_defect_visible is None or dvis is None or dvis >= min_defect_visible or occlusion <= 0:
                break
            resamples += 1
            if resamples >= max_resample:
                break
    obs = observed_in_E(G, size, size, occ_I)
    dvis = float(obs[defect_mask_E].mean()) if defect_mask_E is not None and defect_mask_E.any() else None
    I = I.copy()
    I[occ_I] = 0.5                       # окклюдер — плоский серый прямоугольник
    I = png_roundtrip(I)
    meta = dict(occ_meta or {})
    meta.update(brightness=brightness, noise_sigma=noise_sigma, occ_resamples=resamples)
    return Observation(I, G, np.linalg.inv(G), from_E & ~occ_I, occ_I, float(obs.mean()), dvis, meta)


def factor_cases(seed: int = 2026) -> list[dict]:
    """Однофакторная сетка + фиксированный комбинированный опыт (без полного
    декартова произведения всех сеток)."""
    cases = []
    for factor, values in FACTOR_GRID.items():
        for v in values:
            seeds = range(PERSPECTIVE_SEEDS) if factor == "perspective" else range(1)
            for s in seeds:
                cases.append({"factor": factor, "value": v, "seed": seed + s, "rotation": 0.0, "scale": 1.0,
                              "brightness": 0.0, "occlusion": v if factor == "occlusion" else 0.0})
    for rot, sc, b in itertools.product((15.0, -15.0), (0.75, 1.25), (-0.1, 0.1)):
        cases.append({"factor": "combined", "value": f"r{rot:+g}_s{sc:g}_b{b:+g}", "seed": seed,
                      "rotation": rot, "scale": sc, "brightness": b, "occlusion": 0.10})
    for i, c in enumerate(cases):
        c["case_id"] = f"{c['factor']}_{c['value']}_{c['seed']}".replace(".", "p")
        c["case_index"] = i
    return cases


def case_matrix(case: dict, size, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    f, v = case["factor"], case["value"]
    if f == "rotation":
        return rotation_matrix(v, size), 1
    if f == "scale":
        return scale_matrix(v, size), 1
    if f == "shear":
        return shear_matrix(v, size), 1
    if f == "perspective":
        return (np.eye(3), 1) if v == 0 else perspective_matrix(v, size, rng)
    if f == "occlusion":
        return np.eye(3), 1
    if f == "combined":
        return scale_matrix(case["scale"], size) @ rotation_matrix(case["rotation"], size), 1
    raise ValueError(f)


def generate_case(E: np.ndarray, case: dict, **kw) -> Observation:
    rng = np.random.default_rng(case["seed"])
    size = (E.shape[1], E.shape[0])
    G, attempts = case_matrix(case, size, rng)
    obs = observe(E, G, rng, brightness=case["brightness"], occlusion=case["occlusion"], **kw)
    obs.meta.update(case_id=case["case_id"], factor=case["factor"], value=case["value"],
                    seed=case["seed"], attempts=attempts, G=G.tolist(), visibility=obs.visibility)
    return obs


# ---------------------------------------------------------------- дефекты O(E)

DEFECT_AREAS = (0.005, 0.02, 0.05)
COLOR_SHIFTS = (5.0, 10.0, 20.0)


def defect_specs() -> list[dict]:
    specs = []
    for area in DEFECT_AREAS:
        for ch in ("a", "b"):
            for s in COLOR_SHIFTS:
                specs.append({"type": "color", "area": area, "channel": ch, "shift": s})
        specs.append({"type": "scratch", "area": area})
        specs.append({"type": "paste", "area": area})
    return specs


def central_box(size, frac: float = 0.6) -> tuple[float, float, float, float]:
    W, H = size
    mx, my = (1 - frac) / 2 * W, (1 - frac) / 2 * H
    return mx, my, W - mx, H - my


def _disk_mask(size, area_px: float, box, rng) -> np.ndarray:
    W, H = size
    r = np.sqrt(area_px / np.pi)
    x0, y0, x1, y1 = box
    cx = rng.uniform(x0 + r, x1 - r)
    cy = rng.uniform(y0 + r, y1 - r)
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _segment_dist(size, p0, p1) -> np.ndarray:
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.asarray(p1) - np.asarray(p0)
    t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(xx - (p0[0] + t * d[0]), yy - (p0[1] + t * d[1]))


def _scratch_mask(size, area_px: float, box, rng, max_attempts: int = 1000) -> tuple[np.ndarray, dict]:
    """Отрезок случайной ориентации внутри центральной области; ширина
    подбирается бисекцией до заданной дискретной площади."""
    W, H = size
    x0, y0, x1, y1 = box
    L = 0.5 * min(x1 - x0, y1 - y0)
    for _ in range(max_attempts):
        c = rng.uniform([x0, y0], [x1, y1])
        th = rng.uniform(0, np.pi)
        h = 0.5 * L * np.array([np.cos(th), np.sin(th)])
        p0, p1 = c - h, c + h
        if all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (p0, p1)):
            break
    dist = _segment_dist(size, p0, p1)
    lo, hi = 0.0, float(min(W, H))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (dist <= mid / 2).sum() < area_px:
            lo = mid
        else:
            hi = mid
    w = hi if abs((dist <= hi / 2).sum() - area_px) <= abs((dist <= lo / 2).sum() - area_px) else lo
    return dist <= w / 2, {"p0": p0.tolist(), "p1": p1.tolist(), "width": w}


def apply_defect(E: np.ndarray, spec: dict, rng: np.random.Generator,
                 donor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """O_δ(E): (изменённое изображение, маска изменения в координатах E, info).
    Дефект всегда в центральных 60% ROI — вне опорной полосы 10% (Г2-116)."""
    H, W = E.shape[:2]
    size = (W, H)
    area_px = spec["area"] * W * H
    box = central_box(size)
    out = E.copy()
    info: dict = {}
    if spec["type"] == "color":
        mask = _disk_mask(size, area_px, box, rng)
        lab = srgb_to_lab(E[mask])
        lab[:, 1 if spec["channel"] == "a" else 2] += spec["shift"]     # L* сохраняется
        out[mask] = lab_to_srgb(lab)                                   # обратное преобразование + clip
    elif spec["type"] == "scratch":
        mask, info = _scratch_mask(size, area_px, box, rng)
        lab = srgb_to_lab(E[mask])
        dl = 25.0 if lab[:, 0].mean() < 60 else -25.0
        lab[:, 0] = np.clip(lab[:, 0] + dl, 0, 100)
        out[mask] = lab_to_srgb(lab)
        info["dL"] = dl
    elif spec["type"] == "paste":
        if donor is None:
            raise ValueError("paste defect needs a donor image of the same category")
        side = int(round(np.sqrt(area_px)))
        x0, y0, x1, y1 = box
        px = int(rng.integers(int(np.ceil(x0)), int(np.floor(x1)) - side + 1))
        py = int(rng.integers(int(np.ceil(y0)), int(np.floor(y1)) - side + 1))
        Hd, Wd = donor.shape[:2]
        dx = int(rng.integers(0, Wd - side + 1))
        dy = int(rng.integers(0, Hd - side + 1))
        mask = np.zeros((H, W), dtype=bool)
        mask[py:py + side, px:px + side] = True
        out[mask] = donor[dy:dy + side, dx:dx + side].reshape(-1, donor.shape[2])
        info.update(dst=[px, py, side], src=[dx, dy])
    else:
        raise ValueError(spec["type"])
    area = int(mask.sum())
    de = delta_e00(srgb_to_lab(out[mask]), srgb_to_lab(E[mask])) if area else np.zeros(0)
    info.update(spec, actual_area_px=area, actual_area_frac=area / (W * H),
                mean_dE00=float(de.mean()) if area else 0.0)
    return out, mask, info
