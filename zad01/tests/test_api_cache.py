"""Публичный API (раздел 6) и кеш представлений."""

import numpy as np
import pytest

import photometric
from photometric.cache import LazyImage, RepresentationCache
from photometric.io import save_png
from photometric.methods import DEFAULT_METHOD_IDS


def test_public_api(tmp_path, img_a, img_b):
    save_png(img_a, tmp_path / "q.png")
    save_png(img_b, tmp_path / "r.png")
    img = photometric.prepare_image(tmp_path / "q.png", "sRGB")
    assert img.dtype == np.float64 and img.shape == img_a.shape
    r1 = photometric.represent(img, "zncc")
    r2 = photometric.represent(photometric.prepare_image(tmp_path / "r.png"), "zncc")
    assert photometric.compare(r1, r2).status == "OK"

    rows = photometric.score_pair({"query_path": tmp_path / "q.png", "reference_path": tmp_path / "q.png"}, {"methods": DEFAULT_METHOD_IDS})
    assert {r["method_id"] for r in rows} == set(DEFAULT_METHOD_IDS)
    by = {r["method_id"]: r for r in rows}
    assert by["mse"]["score"] == 0.0 and by["mse"]["raw_value"] == 0.0  # score = −D
    assert by["hist_rgb8"]["score"] == pytest.approx(1.0)


def test_disk_cache_reused_and_invalidated(tmp_path, img_a, img_b):
    p = tmp_path / "a.png"
    save_png(img_a, p)
    c1 = RepresentationCache(tmp_path / "cache")
    rep1, ns1 = c1.get("phash64", "a", LazyImage(p), persist=True)
    c2 = RepresentationCache(tmp_path / "cache")
    rep2, ns2 = c2.get("phash64", "a", LazyImage(p), persist=True)
    assert c2.stats["disk_hits"] == 1 and ns1 == ns2
    np.testing.assert_array_equal(rep1.data, rep2.data)
    # изменение исходного файла инвалидирует запись
    save_png(img_b, p)
    import os, time

    os.utime(p, ns=(time.time_ns(), time.time_ns() + 10**9))
    c3 = RepresentationCache(tmp_path / "cache")
    c3.get("phash64", "a", LazyImage(p), persist=True)
    assert c3.stats["computed"] == 1


def test_pixel_query_not_persisted(tmp_path, img_a):
    p = tmp_path / "a.png"
    save_png(img_a, p)
    c = RepresentationCache(tmp_path / "cache")
    c.get("mse", "q", LazyImage(p), persist=False)
    assert not list((tmp_path / "cache").rglob("*.npz")) if (tmp_path / "cache").exists() else True
