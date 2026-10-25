"""ПРИЁМ-Ф2: аффинная инвариантность ZNCC, MSE = b², защита от смешения шкал."""

import numpy as np
import pytest
from skimage.color import deltaE_ciede2000, rgb2lab

from photometric import compare, represent
from photometric.methods import pixel
from photometric.representations import luminance


def _cmp(method, a, b):
    return compare(represent(a, method), represent(b, method))


@pytest.mark.parametrize("a,b", [(1.2, 0.05), (0.5, 0.3), (2.0, -0.1), (0.8, 0.0)])
def test_zncc_affine(img_a, a, b):
    y = 0.25 + 0.3 * img_a  # диапазон [0.265, 0.535]: x = a·y + b остаётся в [0,1] без clip
    x = a * y + b
    assert 0 <= x.min() and x.max() <= 1
    res = _cmp("zncc", x, y)
    assert res.status == "OK" and res.value == pytest.approx(1.0, abs=1e-9)
    # непосредственно на яркостных векторах
    assert pixel.zncc(a * luminance(y) + b, luminance(y)).value == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("b", [0.01, 0.05, 0.1, -0.07])
def test_mse_shift(img_a, b):
    y = 0.2 + 0.5 * img_a
    x = y + b
    assert _cmp("mse", x, y).value == pytest.approx(b * b, rel=1e-9)


def test_scale_255_rejected(img_a):
    for m in ["mse", "ciede2000_mean", "ncc", "hist_rgb8", "phash64"]:
        with pytest.raises(ValueError):
            represent(img_a * 255.0, m)


def test_mse_and_deltae_depend_on_scale(img_a, img_b):
    """При смене шкалы [0,1] → [0,255] MSE меняется в 255² раз, ΔE00 меняется произвольно:
    смешение единиц обнаруживается, а represent() шкалу [0,255] отвергает."""
    mse01 = pixel.mse(img_a, img_b).value
    mse255 = pixel.mse(img_a * 255, img_b * 255).value
    assert mse255 == pytest.approx(mse01 * 255**2, rel=1e-9)

    de01 = deltaE_ciede2000(rgb2lab(img_a), rgb2lab(img_b)).mean()
    with np.errstate(all="ignore"):
        de255 = deltaE_ciede2000(rgb2lab(img_a * 255), rgb2lab(img_b * 255)).mean()
    assert abs(de255 - de01) > 1.0
    assert _cmp("ciede2000_mean", img_a, img_b).value == pytest.approx(de01, rel=1e-12)
