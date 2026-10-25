"""ФОТО-03 / Ф-1: порядок операций, контроли, шум, JPEG, перестановка."""

import numpy as np
import pytest

from photometric import representations as R
from photometric import synthetic as S
from photometric.io import prepare_image, save_png


def test_f1_order_linear_clip_gamma():
    img = np.array([[[0.1, 0.5, 0.95]]])
    a, b, g = 1.4, 0.2, 0.6
    expected = np.clip(a * (img - 0.5) + 0.5 + b, 0, 1) ** g
    np.testing.assert_allclose(S.photometric_transform(img, a, b, g), expected)
    swapped = np.clip(a * (img**g - 0.5) + 0.5 + b, 0, 1)  # γ до линейного — неверный порядок
    assert not np.allclose(S.photometric_transform(img, a, b, g), swapped)


def test_contrast_center_and_brightness_clip():
    img = np.full((1, 1, 3), 0.5)
    assert np.all(S.photometric_transform(img, a=1.4) == 0.5)  # центр 0.5 неподвижен
    np.testing.assert_allclose(S.photometric_transform(np.full((1, 1, 3), 0.9), b=0.2), 1.0)


def test_controls_are_identity(img_a):
    np.testing.assert_array_equal(S.photometric_transform(img_a, 1.0, 0.0, 1.0), np.clip(img_a, 0, 1))
    np.testing.assert_array_equal(S.gaussian_noise(img_a, 0.0, np.random.default_rng(0)), img_a)


def test_plan_single_factors():
    e2 = dict(brightness=[-0.2, 0.2], contrast=[0.6], gamma=[1.6], noise_sigma=[0.0, 0.1], noise_seeds=5, noise_seed=1,
              jpeg_quality=[90], jpeg={"library": "pillow", "subsampling": "4:4:4", "optimize": False})
    items = S.e2_plan(["I01"], e2)
    assert len(items) == 2 + 1 + 1 + 2 * 5 + 1
    for it in items:
        if it["factor"] == "brightness":
            assert it["a"] == 1.0 and it["gamma"] == 1.0
        if it["factor"] == "gamma":
            assert it["a"] == 1.0 and it["b"] == 0.0
    assert len({it["query_id"] for it in items}) == len(items)


def test_noise_deterministic_and_seed_dependent(img_a):
    r1 = S.gaussian_noise(img_a, 0.05, S.noise_rng(7, "I03", 0.05, 0))
    r2 = S.gaussian_noise(img_a, 0.05, S.noise_rng(7, "I03", 0.05, 0))
    r3 = S.gaussian_noise(img_a, 0.05, S.noise_rng(7, "I03", 0.05, 1))
    np.testing.assert_array_equal(r1, r2)
    assert not np.array_equal(r1, r3)
    # независимый шум по каналам
    d = r1 - img_a
    assert abs(np.corrcoef(d[..., 0].ravel(), d[..., 1].ravel())[0, 1]) < 0.1


def test_jpeg_fixed_settings(img_a):
    j90 = S.jpeg_roundtrip(img_a, 90)
    j30 = S.jpeg_roundtrip(img_a, 30)
    np.testing.assert_array_equal(j90, S.jpeg_roundtrip(img_a, 90))
    assert np.abs(j30 - img_a).mean() > np.abs(j90 - img_a).mean()


def test_permutation_preserves_histogram(img_a):
    p = S.permute_pixels(img_a, np.random.default_rng(0))
    np.testing.assert_array_equal(R.joint_rgb_histogram(p), R.joint_rgb_histogram(img_a))
    assert np.any(p != img_a)


def test_png_roundtrip_exact_for_8bit(tmp_path, img_a):
    q = np.rint(img_a * 255) / 255
    save_png(q, tmp_path / "x.png")
    np.testing.assert_array_equal(prepare_image(tmp_path / "x.png"), q)


def test_rgb_not_bgr(tmp_path):
    img = np.zeros((2, 2, 3))
    img[..., 0] = 1.0  # красный
    save_png(img, tmp_path / "r.png")
    out = prepare_image(tmp_path / "r.png")
    assert out[0, 0, 0] == 1.0 and out[0, 0, 2] == 0.0 and out.dtype == np.float64


def test_materialize_reuses_and_stores_params(tmp_path, img_a):
    save_png(img_a, tmp_path / "I01.png")
    e2 = dict(brightness=[0.1], contrast=[], gamma=[], noise_sigma=[], noise_seeds=1, noise_seed=1, jpeg_quality=[],
              jpeg={"library": "pillow", "subsampling": "4:4:4", "optimize": False})
    items = S.e2_plan(["I01"], e2)
    out = tmp_path / "syn"
    df = S.materialize(items, {"I01": str(tmp_path / "I01.png")}, out, lambda it, img: S.render_e2(it, img, e2))
    png = df["query_path"].iloc[0]
    meta = (out / "I01" / "S_I01_brightness_0.1.json").read_text()
    assert '"synthetic": true' in meta and '"b": 0.1' in meta
    mtime = (out / "I01" / "S_I01_brightness_0.1.png").stat().st_mtime_ns
    S.materialize(items, {"I01": str(tmp_path / "I01.png")}, out, lambda it, img: S.render_e2(it, img, e2))
    assert (out / "I01" / "S_I01_brightness_0.1.png").stat().st_mtime_ns == mtime
    assert df["synthetic"].all() and png.endswith(".png")
