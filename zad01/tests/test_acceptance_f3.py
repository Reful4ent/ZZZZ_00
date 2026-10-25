"""ПРИЁМ-Ф3: контрольные пары CIEDE2000 Sharma–Wu–Dalal, допуск 1e-4 к опубликованным значениям."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.color import deltaE_ciede2000

from photometric import compare, represent

FIXTURE = Path(__file__).parent / "fixtures" / "ciede2000_sharma.csv"


def load():
    return pd.read_csv(FIXTURE, comment="#")


def test_fixture_complete():
    df = load()
    assert len(df) == 34 and df["pair"].tolist() == list(range(1, 35))


def test_sharma_pairs_published_values():
    df = load()
    lab1 = df[["L1", "a1", "b1"]].to_numpy(np.float64)
    lab2 = df[["L2", "a2", "b2"]].to_numpy(np.float64)
    got = deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1)
    # Опубликованные значения округлены до 4 знаков: |got − pub| ≤ 0.5e-4 < 1e-4.
    np.testing.assert_allclose(got, df["dE00"].to_numpy(), rtol=0, atol=1e-4)
    # Симметрия формулы
    np.testing.assert_allclose(deltaE_ciede2000(lab2, lab1), df["dE00"].to_numpy(), rtol=0, atol=1e-4)


def test_same_color_zero():
    df = load()
    lab = df[["L1", "a1", "b1"]].to_numpy(np.float64)
    assert np.all(deltaE_ciede2000(lab, lab) == 0)


def test_method_uses_same_backend_and_params():
    """ciede2000_mean — это среднее попиксельных ΔE00 после rgb2lab(D65, 2°), а не ΔE76 и не ΔE средних цветов."""
    from skimage.color import deltaE_cie76, rgb2lab

    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1, (8, 8, 3))
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1)
    v = compare(represent(a, "ciede2000_mean"), represent(b, "ciede2000_mean")).value
    la, lb = rgb2lab(a, illuminant="D65", observer="2"), rgb2lab(b, illuminant="D65", observer="2")
    assert v == pytest.approx(deltaE_ciede2000(la, lb).mean(), rel=1e-12)
    assert abs(v - deltaE_cie76(la, lb).mean()) > 1e-3
    mean_color = deltaE_ciede2000(la.reshape(-1, 3).mean(0), lb.reshape(-1, 3).mean(0))
    assert abs(v - mean_color) > 1e-3
