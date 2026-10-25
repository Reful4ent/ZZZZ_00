"""Г2-167: выбор порога, тай-брейки, fallback, правило ≥, undefined в знаменателях."""

import math

import numpy as np
import pytest

from photometric.thresholds import select_threshold


def test_basic_and_ge_rule():
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    y = [1, 1, 0, 1, 0, 0]
    r = select_threshold(s, y, alpha=0.05)
    # FPR ≤ 0.05 при 3 отрицательных → FP = 0 → max recall при t = 0.8 (оба ≥ 0.8)
    assert r["threshold"] == 0.8 and r["recall_all"] == pytest.approx(2 / 3) and r["fpr_all"] == 0


def test_alpha_allows_fp():
    s = np.arange(40, dtype=float)
    y = np.array([0] * 20 + [1] * 20)
    y[19] = 1
    y[20] = 0  # один отрицательный среди высоких
    r = select_threshold(s, y, alpha=0.05)  # 20 отрицательных → допускается 1 FP
    assert r["fp"] == 1 and r["recall_all"] == 1.0 and r["threshold"] == 19.0


def test_tie_break_smaller_fpr_then_stricter():
    # t=0.5 и t=0.7: recall одинаков (1.0), FPR 0 у обоих → более строгий 0.7
    s = [0.9, 0.7, 0.5, 0.1]
    y = [1, 1, 0, 0]
    r = select_threshold(s, y, alpha=0.6)
    assert r["threshold"] == 0.7 and r["recall_all"] == 1.0 and r["fpr_all"] == 0.0
    # recall равен, FPR различается → меньший FPR
    s = [0.9, 0.8, 0.7, 0.6, 0.3]
    y = [1, 0, 1, 0, 0]
    r = select_threshold(s, y, alpha=0.7)
    assert r["threshold"] == 0.7 and r["fpr_all"] == pytest.approx(1 / 3)


def test_fallback_above_max():
    s = [0.9, 0.5, 0.4]
    y = [0, 1, 1]  # самый высокий — отрицательный: любое непустое множество даёт FPR = 1
    r = select_threshold(s, y, alpha=0.05)
    assert r["fallback_above_max"] and r["threshold"] > 0.9 and r["recall_all"] == 0 and r["fpr_all"] == 0
    assert r["note"]


def test_undefined_counted_in_denominators():
    s = [0.9, math.nan, 0.2, math.nan]
    y = [1, 1, 0, 0]
    r = select_threshold(s, y, alpha=0.05)
    assert r["recall_all"] == 0.5 and r["coverage"] == 0.5 and r["n_pos"] == 2


def test_ties_in_scores():
    s = [0.5, 0.5, 0.5, 0.1]
    y = [1, 0, 1, 0]
    r = select_threshold(s, y, alpha=0.05)
    # Нельзя отделить отрицательный 0.5 от положительных → fallback
    assert r["fallback_above_max"]


def test_distance_scores_negated():
    # для расстояний score = −D: меньшая дистанция → больший score
    d = np.array([0.01, 0.02, 0.3, 0.4])
    y = [1, 1, 0, 0]
    r = select_threshold(-d, y, alpha=0.05)
    assert r["threshold"] == -0.02 and r["recall_all"] == 1.0
