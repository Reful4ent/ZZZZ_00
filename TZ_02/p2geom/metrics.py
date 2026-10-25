"""Метрики оценщика (раздел 8). Используют GT; конвейеру не передаются."""
from __future__ import annotations

import numpy as np

from .homography import project, transfer_errors


def p_corr(x: np.ndarray, u: np.ndarray, H_gt: np.ndarray, thr: float = 3.0) -> tuple[float | None, int]:
    """Г-3: доля пар M0 с e_gt <= thr. Пустое M0 -> (None, 0)."""
    if len(x) == 0:
        return None, 0
    e = transfer_errors(H_gt, x, u)
    n = int((e <= thr).sum())
    return n / len(x), n


def eval_grid(size_I: tuple[int, int], H_gt: np.ndarray, size_E: tuple[int, int], n: int = 10) -> np.ndarray:
    """G: сетка n×n по I; точки, GT-проекция которых вне E, исключаются."""
    W, H = size_I
    xs, ys = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, n))
    g = np.column_stack([xs.ravel(), ys.ravel()])
    p, ok = project(H_gt, g)
    We, He = size_E
    inside = ok & (p[:, 0] >= -0.5) & (p[:, 0] <= We - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= He - 0.5)
    return g[inside]


def homography_error(H_hat: np.ndarray | None, H_gt: np.ndarray, size_I, size_E, n: int = 10) -> float | None:
    """Г-4: средняя ошибка переноса сетки. None — нет H или пустая сетка; inf —
    π(Ĥx) не определена хотя бы в одной точке."""
    if H_hat is None:
        return None
    g = eval_grid(size_I, H_gt, size_E, n)
    if len(g) == 0:
        return None
    ph, okh = project(H_hat, g)
    pg, _ = project(H_gt, g)
    if not okh.all():
        return float("inf")
    return float(np.linalg.norm(ph - pg, axis=1).mean())


def success_at(e_H: list[float | None], thresholds=(1.0, 3.0, 5.0)) -> dict[float, float]:
    """Доля успехов от всех пар; отсутствие H -> неуспех."""
    arr = np.array([np.inf if e is None else e for e in e_H], dtype=float)
    if len(arr) == 0:
        return {t: float("nan") for t in thresholds}
    return {t: float((arr <= t).mean()) for t in thresholds}


def repeatability(kp_I: np.ndarray, kp_E: np.ndarray, H_gt: np.ndarray, size_I, size_E,
                  thr: float = 3.0) -> tuple[float | None, int, int, int]:
    """Г-5: Rep = N_rep / min(N_I^vis, N_E^vis); взаимно однозначное greedy-связывание
    по возрастанию расстояния (tie-break по индексам I, затем E). Нулевой знаменатель -> None."""
    We, He = size_E
    Wi, Hi = size_I
    pI, okI = project(H_gt, kp_I)
    visI = okI & (pI[:, 0] >= -0.5) & (pI[:, 0] <= We - 0.5) & (pI[:, 1] >= -0.5) & (pI[:, 1] <= He - 0.5)
    pE, okE = project(np.linalg.inv(H_gt), kp_E)
    visE = okE & (pE[:, 0] >= -0.5) & (pE[:, 0] <= Wi - 0.5) & (pE[:, 1] >= -0.5) & (pE[:, 1] <= Hi - 0.5)
    nI, nE = int(visI.sum()), int(visE.sum())
    den = min(nI, nE)
    if den == 0:
        return None, 0, nI, nE
    iI, iE = np.nonzero(visI)[0], np.nonzero(visE)[0]
    D = np.linalg.norm(pI[iI][:, None, :] - kp_E[iE][None, :, :], axis=-1)
    a, b = np.nonzero(D <= thr)
    order = np.lexsort((iE[b], iI[a], D[a, b]))
    usedI, usedE = set(), set()
    n_rep = 0
    for k in order:
        i, j = iI[a[k]], iE[b[k]]
        if i in usedI or j in usedE:
            continue
        usedI.add(i)
        usedE.add(j)
        n_rep += 1
    return n_rep / den, n_rep, nI, nE


def inspection_metrics(y_true, y_pred) -> dict:
    """Положительный класс обнаружения — изменение (y_true = 0).
    y_pred ∈ {1, 0, None}; None — отказ ⊥. «_all» — от всех пар, включая отказы."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.array([-1 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v) for v in y_pred])
    m = {}
    for t in (1, 0):
        for p, name in ((1, "1"), (0, "0"), (-1, "reject")):
            m[f"n_true{t}_pred{name}"] = int(((yt == t) & (yp == p)).sum())
    n0, n1 = int((yt == 0).sum()), int((yt == 1).sum())
    n = len(yt)
    m["n"] = n
    m["recall_all"] = m["n_true0_pred0"] / n0 if n0 else float("nan")
    m["fpr_all"] = m["n_true1_pred0"] / n1 if n1 else float("nan")
    m["coverage"] = float((yp != -1).mean()) if n else float("nan")
    m["false_accept_rate_all"] = m["n_true0_pred1"] / n0 if n0 else float("nan")      # пропуск изменения
    m["false_reject_rate_all"] = m["n_true1_pred0"] / n1 if n1 else float("nan")      # ложное отклонение
    acc = yp != -1
    m["selective_accuracy"] = float((yp[acc] == yt[acc]).mean()) if acc.any() else float("nan")
    return m
