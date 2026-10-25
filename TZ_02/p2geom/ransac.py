"""estimate_homography: обязательная ветвь RANSAC (Г2-105…Г2-107) и OpenCV baseline.

Обязательная ветвь:
* выборки по 4 уникальные пары (np.random.Generator с фиксированным seed);
* вырожденные выборки (коллинеарные тройки, совпадающие точки, несогласованная
  ориентация троек в I и E) отбрасываются и учитываются;
* нормализованный DLT;
* выбор гипотезы лексикографически: больше инлаеров, затем меньше сумма
  квадратов ошибок инлаеров; при полном равенстве остаётся более ранняя попытка;
* лимит max_trials попыток (включая вырожденные); адаптивная остановка по
  целевой вероятности протоколируется (stop_reason, required_trials);
* каждая попытка получает конечный статус в trial_log.
Уточнение: scipy.optimize.least_squares (loss=linear) на фиксированных инлаерах
победившей гипотезы, масштаб фиксирован по наибольшему по модулю элементу.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import least_squares

from .config import RansacConfig
from .homography import dlt_batch, normalize_scale, transfer_errors

TRIAL_SCORED = 0
TRIAL_DEGENERATE = 1
TRIAL_DLT_FAILED = 2
TRIAL_STATUS_NAMES = {TRIAL_SCORED: "SCORED", TRIAL_DEGENERATE: "DEGENERATE_SAMPLE",
                      TRIAL_DLT_FAILED: "DLT_FAILED"}

_TRIPLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_CHUNK = 256


@dataclass
class HomographyEstimate:
    method: str
    status: str                        # OK | NO_MATCHES | TOO_FEW_MATCHES | NO_MODEL
    H: np.ndarray | None = None        # I -> E, исходные координаты
    H0: np.ndarray | None = None       # победившая гипотеза до уточнения
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))   # e_ij(Ĥ) для всех M0
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    refine_status: str = "SKIPPED"     # OK | REFINE_FAILED | SKIPPED
    sse0: float = float("nan")
    sse_refined: float = float("nan")
    n_trials: int = 0
    n_degenerate: int = 0
    n_dlt_failed: int = 0
    best_trial: int = -1
    stop_reason: str = ""              # max_trials | adaptive | not_run
    required_trials: int | None = None
    trial_log: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # trial, status, n_in, sse
    seconds: float = 0.0


def required_trials(inlier_ratio: float, confidence: float, sample_size: int = 4) -> float:
    p = inlier_ratio ** sample_size
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return math.inf
    return math.log(1.0 - confidence) / math.log1p(-p)


def degenerate_samples(xs: np.ndarray, us: np.ndarray, tol: float) -> np.ndarray:
    """(B, 4, 2) -> bool (B,). Коллинеарность/совпадение в любом изображении или
    несогласованная ориентация троек между I и E."""

    def cross_and_scale(p):
        a, b, c = p[:, _TRIPLES[:, 0]], p[:, _TRIPLES[:, 1]], p[:, _TRIPLES[:, 2]]
        v1, v2 = b - a, c - a
        cr = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        sc = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
        return cr, sc

    cx, sx = cross_and_scale(xs)
    cu, su = cross_and_scale(us)
    collinear = np.any(np.abs(cx) <= tol * sx, axis=1) | np.any(np.abs(cu) <= tol * su, axis=1)
    sx_sign, su_sign = np.sign(cx), np.sign(cu)
    same = np.all(sx_sign == su_sign, axis=1)
    flipped = np.all(sx_sign == -su_sign, axis=1)
    return collinear | ~(same | flipped)


def _batch_errors(Hs: np.ndarray, x: np.ndarray, u: np.ndarray, w_eps: float) -> np.ndarray:
    X = np.hstack([x, np.ones((len(x), 1))])
    P = np.einsum("bij,nj->bni", Hs, X)
    w = P[..., 2]
    valid = np.abs(w) > w_eps
    with np.errstate(divide="ignore", invalid="ignore"):
        q = P[..., :2] / w[..., None]
        e = np.linalg.norm(q - u[None], axis=-1)
    return np.where(valid, e, np.inf)


def _ransac_own(x: np.ndarray, u: np.ndarray, cfg: RansacConfig) -> HomographyEstimate:
    n = len(x)
    rng = np.random.default_rng(cfg.seed)
    best_n, best_sse, best_H, best_trial = -1, math.inf, None, -1
    log_rows: list[np.ndarray] = []
    t = 0
    n_degen = n_dlt_fail = 0
    stop_reason, req = "max_trials", None
    done = False
    while t < cfg.max_trials and not done:
        B = min(_CHUNK, cfg.max_trials - t)
        samples = np.stack([rng.choice(n, 4, replace=False) for _ in range(B)])
        xs, us = x[samples], u[samples]
        status = np.where(degenerate_samples(xs, us, cfg.collinear_tol), TRIAL_DEGENERATE, TRIAL_SCORED)
        n_in = np.full(B, -1, dtype=np.int64)
        sse = np.full(B, np.nan)
        Hs = np.full((B, 3, 3), np.nan)
        idx = np.nonzero(status == TRIAL_SCORED)[0]
        if idx.size:
            Hb, ok = dlt_batch(xs[idx], us[idx])
            status[idx[~ok]] = TRIAL_DLT_FAILED
            good = idx[ok]
            if good.size:
                Hs[good] = Hb[ok]
                err = _batch_errors(Hb[ok], x, u, cfg.w_eps)
                inl = err <= cfg.eps_g
                n_in[good] = inl.sum(1)
                sse[good] = (np.where(inl, err, 0.0) ** 2).sum(1)
        executed = 0
        for k in range(B):
            t += 1
            executed += 1
            if status[k] == TRIAL_DEGENERATE:
                n_degen += 1
            elif status[k] == TRIAL_DLT_FAILED:
                n_dlt_fail += 1
            elif n_in[k] > best_n or (n_in[k] == best_n and sse[k] < best_sse):
                best_n, best_sse, best_H, best_trial = int(n_in[k]), float(sse[k]), Hs[k].copy(), t - 1
            if cfg.adaptive and best_n > 0:
                req = required_trials(best_n / n, cfg.confidence)
                if t >= req:
                    stop_reason, done = "adaptive", True
                    break
        log_rows.append(np.column_stack([np.arange(t - executed, t), status[:executed],
                                         n_in[:executed], sse[:executed]]))
    est = HomographyEstimate("own", "OK" if best_H is not None else "NO_MODEL")
    est.n_trials, est.n_degenerate, est.n_dlt_failed = t, n_degen, n_dlt_fail
    est.best_trial, est.stop_reason = best_trial, stop_reason
    est.required_trials = None if req is None or math.isinf(req) else int(math.ceil(req))
    est.trial_log = np.vstack(log_rows) if log_rows else np.zeros((0, 4))
    if best_H is not None:
        est.H0 = normalize_scale(best_H)
        est.sse0 = best_sse
    return est


def refine_homography(H0: np.ndarray, x: np.ndarray, u: np.ndarray, max_nfev: int,
                      admissible: Callable[[np.ndarray], bool]) -> tuple[np.ndarray, str, float, float]:
    """Г2-107: (H, статус, sse до, sse после). При ухудшении суммы, ошибке
    оптимизатора или недопустимой H возвращается исходная H0 и REFINE_FAILED."""
    Hn = normalize_scale(H0)
    k = int(np.argmax(np.abs(Hn)))
    free = np.array([i for i in range(9) if i != k])
    X = np.hstack([x, np.ones((len(x), 1))])

    def build(p):
        h = Hn.ravel().copy()
        h[free] = p
        return h.reshape(3, 3)

    def resid(p):
        P = X @ build(p).T
        return (P[:, :2] / P[:, 2:3] - u).ravel()

    p0 = Hn.ravel()[free]
    sse0 = float(np.sum(resid(p0) ** 2))
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            res = least_squares(resid, p0, loss="linear", max_nfev=max_nfev, x_scale="jac", method="trf")
        H1 = build(res.x)
        r1 = resid(res.x)
        sse1 = float(np.sum(r1 ** 2))
    except Exception:
        return Hn, "REFINE_FAILED", sse0, float("nan")
    if not (np.all(np.isfinite(H1)) and np.all(np.isfinite(r1)) and sse1 <= sse0 and admissible(H1)):
        return Hn, "REFINE_FAILED", sse0, sse1
    return H1, "OK", sse0, sse1


def _opencv_baseline(x: np.ndarray, u: np.ndarray, cfg: RansacConfig) -> HomographyEstimate:
    cv2.setRNGSeed(cfg.seed)
    H, _ = cv2.findHomography(x.astype(np.float64), u.astype(np.float64), cv2.RANSAC,
                              cfg.eps_g, maxIters=cfg.max_trials, confidence=cfg.confidence)
    est = HomographyEstimate("opencv", "OK" if H is not None else "NO_MODEL")
    est.stop_reason = "opencv_internal"
    est.refine_status = "OPENCV_INTERNAL"
    if H is not None:
        est.H0 = normalize_scale(H)
        est.H = est.H0
    return est


def estimate_homography(x: np.ndarray, u: np.ndarray, cfg: RansacConfig,
                        admissible: Callable[[np.ndarray], bool] | None = None) -> HomographyEstimate:
    """x — точки I, u — точки E (исходные координаты, пары M0). Ĥ: I -> E."""
    t0 = time.perf_counter()
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    u = np.asarray(u, dtype=np.float64).reshape(-1, 2)
    n = len(x)
    if n == 0:
        return HomographyEstimate(cfg.estimator, "NO_MATCHES", stop_reason="not_run",
                                  seconds=time.perf_counter() - t0)
    if n < 4:
        return HomographyEstimate(cfg.estimator, "TOO_FEW_MATCHES", stop_reason="not_run",
                                  errors=np.full(n, np.inf), inliers=np.zeros(n, bool),
                                  seconds=time.perf_counter() - t0)
    if cfg.estimator == "own":
        est = _ransac_own(x, u, cfg)
    elif cfg.estimator == "opencv":
        est = _opencv_baseline(x, u, cfg)
    else:
        raise ValueError(f"unknown estimator {cfg.estimator!r}")
    if est.H0 is None:
        est.errors, est.inliers = np.full(n, np.inf), np.zeros(n, bool)
        est.seconds = time.perf_counter() - t0
        return est
    if cfg.estimator == "own":
        inl0 = transfer_errors(est.H0, x, u, cfg.w_eps) <= cfg.eps_g
        if cfg.refine:
            est.H, est.refine_status, est.sse0, est.sse_refined = refine_homography(
                est.H0, x[inl0], u[inl0], cfg.refine_max_nfev, admissible or (lambda H: True))
        else:
            est.H = est.H0
    # Инлаеры пересчитываются один раз по итоговой H.
    est.errors = transfer_errors(est.H, x, u, cfg.w_eps)
    est.inliers = est.errors <= cfg.eps_g
    est.seconds = time.perf_counter() - t0
    return est
