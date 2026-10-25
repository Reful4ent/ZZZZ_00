"""Геометрические примитивы: проекция π, нормализованный DLT, пересчёт систем
координат (Г2-102), проверка допустимости H (H_adm), покрытие (Г2-109).

Соглашение о координатах: (x, y) в пикселях, центр пикселя (c, r) имеет
координаты (c, r) — как в OpenCV. Прямоугольник изображения W×H —
[-0.5, W-0.5] × [-0.5, H-0.5], его площадь W·H.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- проекция


def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.hstack([pts, np.ones((len(pts), 1))])


def project(H: np.ndarray, pts: np.ndarray, w_eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """π(H x̃) (Г-1). Возвращает (точки N×2, valid). При |w| <= w_eps точка
    не нормализуется: координаты NaN, valid=False."""
    xh = to_homogeneous(pts) @ np.asarray(H, dtype=np.float64).T
    w = xh[:, 2]
    valid = np.abs(w) > w_eps
    out = np.full((len(xh), 2), np.nan)
    out[valid] = xh[valid, :2] / w[valid, None]
    return out, valid


def transfer_errors(H: np.ndarray, x: np.ndarray, u: np.ndarray, w_eps: float = 1e-12) -> np.ndarray:
    """e_ij(H) = ||u_j − π(H x̃_i)||₂ (Г2-104); inf там, где π не определена."""
    p, valid = project(H, x, w_eps)
    err = np.full(len(p), np.inf)
    err[valid] = np.linalg.norm(np.asarray(u, dtype=np.float64)[valid] - p[valid], axis=1)
    return err


# ---------------------------------------------------------------- DLT


def hartley_matrix(pts: np.ndarray) -> np.ndarray:
    """Нормализация Хартли: центр в 0, среднее расстояние √2."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _dlt_rows(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Матрица DLT (…, 2N, 9) для пар x -> u (последняя ось — координаты)."""
    X, Y = x[..., 0], x[..., 1]
    U, V = u[..., 0], u[..., 1]
    z = np.zeros_like(X)
    o = np.ones_like(X)
    r1 = np.stack([-X, -Y, -o, z, z, z, U * X, U * Y, U], axis=-1)
    r2 = np.stack([z, z, z, -X, -Y, -o, V * X, V * Y, V], axis=-1)
    rows = np.stack([r1, r2], axis=-2)                  # (..., N, 2, 9)
    return rows.reshape(*rows.shape[:-3], -1, 9)


def dlt_batch(x: np.ndarray, u: np.ndarray, rank_tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Нормализованный DLT для батча (B, N, 2) -> (B, 3, 3), ok (B,).
    ok=False, если система вырождена (второе наименьшее сингулярное число ~0)."""
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    B = x.shape[0]
    Tx = np.stack([hartley_matrix(x[b]) for b in range(B)])
    Tu = np.stack([hartley_matrix(u[b]) for b in range(B)])
    xn = np.einsum("bij,bnj->bni", Tx, np.concatenate([x, np.ones(x.shape[:2] + (1,))], -1))[..., :2]
    un = np.einsum("bij,bnj->bni", Tu, np.concatenate([u, np.ones(u.shape[:2] + (1,))], -1))[..., :2]
    A = _dlt_rows(xn, un)
    _, s, vt = np.linalg.svd(A)
    h = vt[:, -1, :].reshape(B, 3, 3)
    # Решение единственно (ядро одномерно), если восьмое сингулярное число > 0;
    # при N > 4 девятое — остаток МНК.
    ok = s[:, 7] > rank_tol * s[:, 0]
    H = np.linalg.inv(Tu) @ h @ Tx
    with np.errstate(invalid="ignore", divide="ignore"):
        H = H / np.abs(H).reshape(B, -1).max(axis=1)[:, None, None]
    ok &= np.all(np.isfinite(H.reshape(B, -1)), axis=1)
    return H, ok


def dlt(x: np.ndarray, u: np.ndarray) -> np.ndarray | None:
    H, ok = dlt_batch(np.asarray(x)[None], np.asarray(u)[None])
    return H[0] if ok[0] else None


def normalize_scale(H: np.ndarray) -> np.ndarray:
    """Масштаб по наибольшему по модулю элементу (=1, знак сохраняется через элемент)."""
    k = int(np.argmax(np.abs(H)))
    return H / H.flat[k]


# ---------------------------------------------------------------- системы координат


def resize_crop_matrix(orig_size: tuple[int, int], work_size: tuple[int, int],
                       crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """A: исходные координаты -> рабочие. crop=(x0, y0, w, h) в исходных пикселях,
    выполняется до resize; work_size=(W', H') — размер после resize."""
    x0, y0 = 0, 0
    W, H = orig_size
    if crop is not None:
        x0, y0, W, H = crop
    sx, sy = work_size[0] / W, work_size[1] / H
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    # Центр пикселя: x' = s (x + 0.5) − 0.5 (семантика cv2.resize).
    S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    return S @ T


def work_to_original(H_work: np.ndarray, A_I: np.ndarray, A_E: np.ndarray) -> np.ndarray:
    """Ĥ = A_E⁻¹ Ĥ_work A_I (Г2-102)."""
    return np.linalg.inv(A_E) @ H_work @ A_I


def original_to_work(H: np.ndarray, A_I: np.ndarray, A_E: np.ndarray) -> np.ndarray:
    return A_E @ H @ np.linalg.inv(A_I)


def coord_normalizer(size: tuple[int, int]) -> np.ndarray:
    """N(W,H): x' = 2x/(W−1) − 1, y' = 2y/(H−1) − 1."""
    W, H = size
    if W <= 1 or H <= 1:
        raise ValueError(f"image size {size}: dimensions <= 1 px are not admissible")
    return np.array([[2.0 / (W - 1), 0, -1.0], [0, 2.0 / (H - 1), -1.0], [0, 0, 1.0]])


def image_corners(size: tuple[int, int]) -> np.ndarray:
    W, H = size
    return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)


# ---------------------------------------------------------------- многоугольники


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Монотонная цепь Эндрю; результат против часовой стрелки (в осях x→, y↑)."""
    pts = np.unique(np.asarray(pts, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def clip_polygon_to_rect(poly: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Сазерленд–Ходжмен: отсечение по прямоугольнику (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    out = [tuple(p) for p in poly]
    edges = [
        (lambda p: p[0] >= x0, lambda a, b: _isect_x(a, b, x0)),
        (lambda p: p[0] <= x1, lambda a, b: _isect_x(a, b, x1)),
        (lambda p: p[1] >= y0, lambda a, b: _isect_y(a, b, y0)),
        (lambda p: p[1] <= y1, lambda a, b: _isect_y(a, b, y1)),
    ]
    for inside, isect in edges:
        if not out:
            break
        src, out = out, []
        prev = src[-1]
        for cur in src:
            if inside(cur):
                if not inside(prev):
                    out.append(isect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(isect(prev, cur))
            prev = cur
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def _isect_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _isect_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def image_rect(size: tuple[int, int]) -> tuple[float, float, float, float]:
    W, H = size
    return (-0.5, -0.5, W - 0.5, H - 0.5)


def hull_coverage(pts: np.ndarray, size: tuple[int, int]) -> float:
    """Area[conv{pts} ∩ R] / Area(R) для прямоугольного ROI R (Г2-109).
    ROI сложной формы не поддерживается: для него нужен отдельный алгоритм."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    hull = convex_hull(pts)
    if len(hull) < 3:
        return 0.0
    clipped = clip_polygon_to_rect(hull, image_rect(size))
    W, H = size
    return abs(polygon_area(clipped)) / float(W * H)


def segments_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def is_simple_quad(q: np.ndarray) -> bool:
    """Четырёхугольник без самопересечения (противоположные стороны не пересекаются)."""
    return not (segments_intersect(q[0], q[1], q[2], q[3]) or segments_intersect(q[1], q[2], q[3], q[0]))


def quad_orientation_signs(q: np.ndarray) -> np.ndarray:
    """Знаки векторных произведений в вершинах четырёхугольника."""
    s = []
    for k in range(4):
        a, b, c = q[k - 1], q[k], q[(k + 1) % 4]
        s.append(np.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    return np.array(s)


# ---------------------------------------------------------------- допустимость H


@dataclass
class Admissibility:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    cond: float = float("nan")
    w_corners: list[float] = field(default_factory=list)
    jac_dets: list[float] = field(default_factory=list)
    area_ratio: float = float("nan")


def jacobian_det(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """det ∂π(Hx)/∂x = det(H) / w³."""
    w = to_homogeneous(pts) @ H[2]
    return np.linalg.det(H) / w ** 3


def check_admissible(H: np.ndarray | None, size_I: tuple[int, int], size_E: tuple[int, int],
                     cond_max: float = 1e10, w_min: float = 1e-8,
                     area_ratio_range: tuple[float, float] | None = None) -> Admissibility:
    """H_adm (раздел 4): конечность, обратимость, cond(N_E H N_I⁻¹) <= cond_max,
    знаменатели в углах ROI I одного знака и |w| >= w_min после нормирования,
    одинаковые положительные знаки якобиана, перенесённый контур без самопересечения."""
    if H is None:
        return Admissibility(False, ["NO_MODEL"])
    H = np.asarray(H, dtype=np.float64)
    reasons: list[str] = []
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return Admissibility(False, ["NON_FINITE"])
    try:
        N_I, N_E = coord_normalizer(size_I), coord_normalizer(size_E)
    except ValueError:
        return Admissibility(False, ["IMAGE_TOO_SMALL"])
    Hn = N_E @ H @ np.linalg.inv(N_I)
    fro = np.linalg.norm(Hn)
    if fro == 0 or not np.isfinite(fro):
        return Admissibility(False, ["SINGULAR"])
    Hn = Hn / fro
    sv = np.linalg.svd(Hn, compute_uv=False)
    if sv[-1] <= 0:
        return Admissibility(False, ["SINGULAR"], cond=float("inf"))
    cond = float(sv[0] / sv[-1])
    if cond > cond_max:
        reasons.append("ILL_CONDITIONED")
    nc = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    w = to_homogeneous(nc) @ Hn[2]
    if not (np.all(w > 0) or np.all(w < 0)):
        reasons.append("DENOMINATOR_SIGN_CHANGE")
    if np.any(np.abs(w) < w_min):
        reasons.append("DENOMINATOR_TOO_SMALL")
    corners = image_corners(size_I)
    center = np.array([[(size_I[0] - 1) / 2.0, (size_I[1] - 1) / 2.0]])
    jd = jacobian_det(H, np.vstack([corners, center]))
    if not np.all(jd > 0):
        reasons.append("JACOBIAN_SIGN")
    area_ratio = float("nan")
    if not reasons:
        q, valid = project(H, corners)
        if not valid.all() or not is_simple_quad(q):
            reasons.append("SELF_INTERSECTION")
        else:
            area_ratio = abs(polygon_area(q)) / abs(polygon_area(corners))
            if area_ratio_range is not None and not (area_ratio_range[0] <= area_ratio <= area_ratio_range[1]):
                reasons.append("SCALE_OUT_OF_RANGE")
    return Admissibility(not reasons, reasons, cond, w.tolist(), jd.tolist(), area_ratio)
