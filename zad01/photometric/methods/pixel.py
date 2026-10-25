"""Попиксельные меры: MSE (Г1-20), NCC (Ф-2), ZNCC (Ф-3).

Размеры и область должны совпадать; resize пары запрещён (ФОТО-04).
Ω — все пиксели изображения (маска допустимых пикселей в ТЗ 01 не задана).
"""

from __future__ import annotations

import math

import numpy as np

from photometric.methods.base import (
    SIZE_MISMATCH,
    UNDEFINED_EPS,
    ZERO_ENERGY,
    ZERO_VARIANCE,
    Result,
)


def mse(i: np.ndarray, e: np.ndarray) -> Result:
    """Г1-20: среднее (I_c − E_c)² по N пикселям и C = 3 каналам, RGB в [0, 1]."""
    if i.shape != e.shape:
        return Result.undefined(SIZE_MISMATCH)
    d = i - e
    return Result(float(np.mean(d * d)))


def ncc(x: np.ndarray, y: np.ndarray, eps: float = UNDEFINED_EPS) -> Result:
    """Ф-2: Σ x_i y_i / sqrt(Σ x_i² · Σ y_i²) — сумма квадратов в каждом множителе."""
    if x.shape != y.shape:
        return Result.undefined(SIZE_MISMATCH)
    x = x.ravel()
    y = y.ravel()
    ex = float(np.dot(x, x))
    ey = float(np.dot(y, y))
    if ex <= eps or ey <= eps:
        return Result.undefined(ZERO_ENERGY)
    return Result(float(np.dot(x, y)) / math.sqrt(ex * ey))


def zncc(x: np.ndarray, y: np.ndarray, eps: float = UNDEFINED_EPS) -> Result:
    """Ф-3: NCC центрированных векторов.

    Undefined (ZERO_VARIANCE), если дисперсия (1/n)Σ(x_i − x̄)² любого вектора ≤ eps;
    для двух одинаковых констант ZNCC = 1 не назначается.
    """
    if x.shape != y.shape:
        return Result.undefined(SIZE_MISMATCH)
    xc = x.ravel() - x.mean()
    yc = y.ravel() - y.mean()
    sx = float(np.dot(xc, xc))
    sy = float(np.dot(yc, yc))
    n = xc.size
    if sx / n <= eps or sy / n <= eps:
        return Result.undefined(ZERO_VARIANCE)
    return Result(float(np.dot(xc, yc)) / math.sqrt(sx * sy))
