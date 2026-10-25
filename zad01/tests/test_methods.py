"""Формулы и однозначные правила методов (раздел 3 ТЗ)."""

import math

import numpy as np
import pytest
import pywt
from scipy.fft import dctn

from photometric import compare, represent
from photometric import representations as R
from photometric.methods import pixel


def test_luminance_weights():
    img = np.zeros((1, 3, 3))
    img[0, 0, 0] = img[0, 1, 1] = img[0, 2, 2] = 1.0
    np.testing.assert_allclose(R.luminance(img)[0], [0.2126, 0.7152, 0.0722])


def test_ncc_sum_of_squares_not_square_of_sum():
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[2.0, 1.0, 4.0]])
    expected = (2 + 2 + 12) / math.sqrt((1 + 4 + 9) * (4 + 1 + 16))
    wrong = (2 + 2 + 12) / math.sqrt((1 + 2 + 3) ** 2 * (2 + 1 + 4) ** 2)
    got = pixel.ncc(x, y).value
    assert got == pytest.approx(expected, rel=1e-15)
    assert abs(got - wrong) > 0.1


def test_zncc_formula():
    rng = np.random.default_rng(0)
    x, y = rng.random(100), rng.random(100)
    assert pixel.zncc(x, y).value == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
    assert pixel.zncc(x, -2 * x + 1).value == pytest.approx(-1.0, abs=1e-12)


def test_size_mismatch_no_resize(img_a):
    small = img_a[:-1]
    for m in ["mse", "ncc", "zncc", "ciede2000_mean"]:
        res = compare(represent(img_a, m), represent(small, m))
        assert res.status == "SIZE_MISMATCH" and math.isnan(res.value)


def test_histogram_bins_and_edge():
    img = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.125, 0.124999, 0.999], [0.5, 0.875, 0.874]]])
    h = R.joint_rgb_histogram(img)
    assert h.shape == (512,) and h.sum() == pytest.approx(1.0, abs=1e-15)
    assert h[0] == 0.25                          # (0,0,0)
    assert h[7 * 64 + 7 * 8 + 7] == 0.25         # значение 1 — последний интервал
    assert h[1 * 64 + 0 * 8 + 7] == 0.25         # 0.125 → bin 1, 0.124999 → bin 0
    assert h[4 * 64 + 7 * 8 + 6] == 0.25


def test_histogram_each_pixel_one_bin(img_a):
    h = R.joint_rgb_histogram(img_a)
    n = img_a.shape[0] * img_a.shape[1]
    np.testing.assert_allclose(h * n, np.rint(h * n), atol=1e-9)
    assert h.sum() == pytest.approx(1.0, abs=1e-12)


def test_hist_intersection_disjoint_and_partial():
    a = np.zeros((2, 2, 3))
    b = np.ones((2, 2, 3))
    assert compare(represent(a, "hist_rgb8"), represent(b, "hist_rgb8")).value == 0.0
    c = a.copy()
    c[0, 0] = 1.0
    assert compare(represent(a, "hist_rgb8"), represent(c, "hist_rgb8")).value == pytest.approx(0.75)


def test_hash_lengths_and_range(img_a, img_b):
    for m in ["ahash64", "dhash64", "phash64", "whash64"]:
        ra, rb = represent(img_a, m), represent(img_b, m)
        assert ra.data.shape == (64,) and ra.data.dtype == bool
        d = compare(ra, rb).value
        assert 0.0 <= d <= 1.0 and d * 64 == int(d * 64)


def test_ahash_rule():
    y = np.tile(np.linspace(0, 1, 8), (8, 1))
    bits = R.ahash_bits(y).reshape(8, 8)
    assert (bits == (y > y.mean())).all()  # строгое «больше», row-major


def test_dhash_rule():
    y = np.tile(np.linspace(0, 1, 9), (8, 1))  # растёт вправо → все биты 1
    assert R.dhash_bits(y).all()
    assert not R.dhash_bits(y[:, ::-1]).any()
    small = R.resize_bilinear(y, 8, 9)
    assert small.shape == (8, 9)


def test_phash_rule_dc_included():
    rng = np.random.default_rng(1)
    y = rng.random((32, 32))
    block = dctn(y, type=2, norm="ortho")[:8, :8]
    expected = (block > np.median(block)).ravel()
    np.testing.assert_array_equal(R.phash_bits(y), expected)
    # ортонормированность: энергия сохраняется
    assert (dctn(y, type=2, norm="ortho") ** 2).sum() == pytest.approx((y**2).sum())


def test_whash_rule_haar_level2():
    rng = np.random.default_rng(2)
    y = rng.random((32, 32))
    ca2 = pywt.wavedec2(y, "haar", mode="periodization", level=2)[0]
    assert ca2.shape == (8, 8)
    # cA2 Haar (ортонормированный) = 4 × среднее блока 4×4
    np.testing.assert_allclose(ca2, 4 * y.reshape(8, 4, 8, 4).mean(axis=(1, 3)), atol=1e-12)
    np.testing.assert_array_equal(R.whash_bits(y), (ca2 > np.median(ca2)).ravel())


def test_resize_is_bilinear_identity_on_same_size():
    rng = np.random.default_rng(3)
    y = rng.random((32, 32))
    np.testing.assert_allclose(R.resize_bilinear(y, 32, 32), y, atol=1e-12)


def test_method_variants_recorded():
    assert represent(np.zeros((4, 4, 3)), "phash64").meta["params"]["variant"] == "dc_included"
    assert represent(np.zeros((4, 4, 3)), "whash64").meta["params"]["variant"] == "haar_level2"


def test_variant_override_changes_config_hash(img_a):
    r1 = represent(img_a, "ahash64")
    r2 = represent(img_a, {"method_id": "ahash64", "resize": {"anti_aliasing": True}})
    assert r1.config_hash != r2.config_hash
    with pytest.raises(ValueError):
        compare(r1, r2)
