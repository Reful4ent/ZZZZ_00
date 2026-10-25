"""ПРИЁМ-Ф1: идентичные непостоянные изображения и undefined-ветви на константах."""

import math

import numpy as np
import pytest

from photometric import compare, represent
from photometric.methods import DEFAULT_METHOD_IDS, ZERO_ENERGY, ZERO_VARIANCE
from photometric.scoring import apply_thresholds, to_score

ATOL = 1e-6
EXPECTED_IDENTICAL = {
    "mse": 0.0,
    "ciede2000_mean": 0.0,
    "ncc": 1.0,
    "zncc": 1.0,
    "hist_rgb8": 1.0,
    "ahash64": 0.0,
    "dhash64": 0.0,
    "phash64": 0.0,
    "whash64": 0.0,
}


def _cmp(method, a, b):
    return compare(represent(a, method), represent(b, method))


def test_all_nine_methods_covered():
    assert set(EXPECTED_IDENTICAL) == set(DEFAULT_METHOD_IDS)


@pytest.mark.parametrize("method", list(EXPECTED_IDENTICAL))
def test_identical_nonconstant(method, img_a):
    res = _cmp(method, img_a, img_a.copy())
    assert res.status == "OK"
    assert res.value == pytest.approx(EXPECTED_IDENTICAL[method], abs=ATOL)


def test_identical_constants_zncc_undefined():
    c = np.full((16, 16, 3), 0.4)
    res = _cmp("zncc", c, c.copy())
    assert res.status == ZERO_VARIANCE and math.isnan(res.value)


def test_zncc_one_constant_undefined(img_a):
    c = np.full_like(img_a, 0.7)
    assert _cmp("zncc", img_a, c).status == ZERO_VARIANCE
    assert _cmp("zncc", c, img_a).status == ZERO_VARIANCE


def test_ncc_nonzero_constants_regular_rule():
    a = np.full((16, 16, 3), 0.4)
    b = np.full((16, 16, 3), 0.9)
    res = _cmp("ncc", a, b)
    assert res.status == "OK" and res.value == pytest.approx(1.0, abs=ATOL)


def test_ncc_zero_energy(img_a):
    z = np.zeros_like(img_a)
    assert _cmp("ncc", z, img_a).status == ZERO_ENERGY
    assert _cmp("ncc", z, z).status == ZERO_ENERGY


def test_other_methods_defined_on_constants():
    a = np.full((16, 16, 3), 0.4)
    for m in ["mse", "hist_rgb8", "ciede2000_mean", "ahash64", "dhash64", "phash64", "whash64"]:
        assert _cmp(m, a, a.copy()).status == "OK"


def test_undefined_is_refusal_not_negative_score():
    import pandas as pd

    assert math.isnan(to_score("similarity", math.nan, ZERO_VARIANCE))
    pred = pd.DataFrame({"method_id": ["zncc", "zncc"], "score": [math.nan, 0.99], "status": [ZERO_VARIANCE, "OK"]})
    out = apply_thresholds(pred, {"zncc": {"threshold": -10.0}})
    assert out["decision"].tolist() == [0, 1]
