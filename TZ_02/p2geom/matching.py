"""match_descriptors: ratio-test в обоих направлениях, затем взаимность (Г2-103).

Порядок соседей — по (расстояние, индекс точки): равные расстояния
разбираются по индексу. Взаимная проверка гарантирует, что каждая точка
входит не более чем в одну пару.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .features import Features


@dataclass
class Matches:
    pairs: np.ndarray        # (K, 2): индекс точки I, индекс точки E
    d1: np.ndarray           # расстояние до лучшего соседа (направление I -> E)
    d2: np.ndarray           # до второго
    ratio: np.ndarray        # r_ij = d1 / d2
    counts: dict
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.pairs)


def distance_matrix(a: np.ndarray, b: np.ndarray, norm: str) -> np.ndarray:
    if norm == "l2":
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return np.sqrt(np.maximum(d2, 0.0))
    if norm == "hamming":
        ab = np.unpackbits(a, axis=1).astype(np.float32)
        bb = np.unpackbits(b, axis=1).astype(np.float32)
        # Целые <= 256: в float32 точно.
        d = ab.sum(1)[:, None] + bb.sum(1)[None, :] - 2.0 * ab @ bb.T
        return d.astype(np.float64)
    raise ValueError(norm)


def _two_nn(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Для каждой строки: индекс лучшего, d1, d2; порядок (расстояние, индекс)."""
    order = np.argsort(D, axis=1, kind="stable")[:, :2]
    rows = np.arange(D.shape[0])
    return order[:, 0], D[rows, order[:, 0]], D[rows, order[:, 1]]


def _ratio_pass(d1: np.ndarray, d2: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    ok = d2 > 0
    r = np.full(len(d1), np.inf)
    r[ok] = d1[ok] / d2[ok]
    return ok & (r < tau), r


def match_descriptors(fI: Features, fE: Features, ratio: float) -> Matches:
    t0 = time.perf_counter()
    nI, nE = len(fI), len(fE)
    counts = {"n_I": nI, "n_E": nE, "fwd_ratio_pass": 0, "bwd_ratio_pass": 0, "mutual": 0}
    empty = Matches(np.zeros((0, 2), int), np.zeros(0), np.zeros(0), np.zeros(0), counts)
    # Меньше двух соседей в направлении — назначение отвергается.
    if nI < 2 or nE < 2 or fI.status != "OK" or fE.status != "OK":
        empty.seconds = time.perf_counter() - t0
        return empty
    D = distance_matrix(fI.desc, fE.desc, fI.norm)
    j_of_i, d1_f, d2_f = _two_nn(D)
    i_of_j, d1_b, d2_b = _two_nn(D.T)
    fwd_ok, r_f = _ratio_pass(d1_f, d2_f, ratio)
    bwd_ok, _ = _ratio_pass(d1_b, d2_b, ratio)
    i_idx = np.nonzero(fwd_ok)[0]
    j_idx = j_of_i[i_idx]
    mutual = bwd_ok[j_idx] & (i_of_j[j_idx] == i_idx)
    i_idx, j_idx = i_idx[mutual], j_idx[mutual]
    counts.update(fwd_ratio_pass=int(fwd_ok.sum()), bwd_ratio_pass=int(bwd_ok.sum()), mutual=int(len(i_idx)))
    return Matches(np.stack([i_idx, j_idx], 1).astype(int), d1_f[i_idx], d2_f[i_idx], r_f[i_idx],
                   counts, time.perf_counter() - t0)
