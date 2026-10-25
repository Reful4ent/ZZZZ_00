"""Приёмочные тесты ПРИЁМ-Г1…ПРИЁМ-Г5 (раздел 9 ТЗ 02)."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import random_h
from p2geom.config import Config, RansacConfig
from p2geom.decision import decide
from p2geom.experiments import run_inspection
from p2geom.features import prepare_work_image
from p2geom.homography import dlt, original_to_work, project, resize_crop_matrix, work_to_original
from p2geom.io import write_png
from p2geom.pipeline import run_pair
from p2geom.ransac import estimate_homography
from p2geom.regions import compare_regions
from p2geom.warp import object_mask, warp_with_mask


# ---------------------------------------------------------------- ПРИЁМ-Г1

def _grid(W, H, n=12):
    xs, ys = np.meshgrid(np.linspace(5, W - 6, n), np.linspace(5, H - 6, n))
    return np.column_stack([xs.ravel(), ys.ravel()])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_g1_noise_free_grid_reproduces_gt(seed):
    rng = np.random.default_rng(seed)
    size = (640, 480)
    H_true = random_h(rng, size, 0.08)            # I -> E
    x = _grid(*size)
    u, ok = project(H_true, x)
    assert ok.all()
    est = estimate_homography(x, u, RansacConfig())
    assert est.status == "OK" and est.refine_status in ("OK", "REFINE_FAILED")
    g = _grid(*size, 20)
    err = np.linalg.norm(project(est.H, g)[0] - project(H_true, g)[0], axis=1)
    assert err.max() <= 1e-4
    # Обратное направление: E -> I.
    est_inv = estimate_homography(u, x, RansacConfig())
    gi = project(H_true, g)[0]
    err_inv = np.linalg.norm(project(est_inv.H, gi)[0] - g, axis=1)
    assert err_inv.max() <= 1e-4


def test_g1_degenerate_and_too_few_do_not_crash():
    x = np.column_stack([np.linspace(0, 100, 30), np.linspace(0, 50, 30)])   # коллинеарные
    u = x + 3.0
    est = estimate_homography(x, u, RansacConfig(max_trials=200))
    assert est.status == "NO_MODEL"
    assert est.n_degenerate + est.n_dlt_failed == est.n_trials == 200
    assert len(est.trial_log) == 200 and set(est.trial_log[:, 1]) <= {1, 2}
    for n in (0, 1, 3):
        est = estimate_homography(np.zeros((n, 2)), np.zeros((n, 2)), RansacConfig())
        assert est.status == ("NO_MATCHES" if n == 0 else "TOO_FEW_MATCHES")
        assert est.H is None


# ---------------------------------------------------------------- ПРИЁМ-Г2

@pytest.mark.parametrize("orig_I,orig_E,crop_I,crop_E,max_side", [
    ((2000, 1500), (1600, 1200), None, None, 1024),
    ((1300, 900), (1000, 1000), (50, 20, 1100, 800), (0, 100, 900, 800), 512),
    ((800, 600), (800, 600), None, None, 1024),
])
def test_g2_work_to_original(orig_I, orig_E, crop_I, crop_E, max_side):
    rng = np.random.default_rng(3)
    H = random_h(rng, orig_I, 0.05)                              # I -> E в исходных координатах
    work = {}
    for name, size, crop in (("I", orig_I, crop_I), ("E", orig_E, crop_E)):
        w, h = (crop[2], crop[3]) if crop else size
        s = min(1.0, max_side / max(w, h))
        work[name] = resize_crop_matrix(size, (int(round(w * s)), int(round(h * s))), crop)
    A_I, A_E = work["I"], work["E"]
    x = _grid(*orig_I, 8)
    u = project(H, x)[0]
    H_work = dlt(project(A_I, x)[0], project(A_E, u)[0])        # оценка в рабочих координатах
    H_back = work_to_original(H_work, A_I, A_E)
    assert np.abs(project(H_back, x)[0] - u).max() < 1e-6
    xw = project(A_I, x)[0]
    assert np.abs(project(original_to_work(H_back, A_I, A_E), xw)[0] - project(H_work, xw)[0]).max() < 1e-6


def test_g2_resize_matrix_matches_cv2_pixel_centres():
    """Точечный источник в исходном изображении попадает в A·x после cv2.resize."""
    W, H = 1600, 1200
    img = np.zeros((H, W, 3))
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = 701.3, 433.7
    img[..., :] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 25.0 ** 2))[..., None]
    gray, A = prepare_work_image(img, 1024)
    g = gray.astype(float)
    g -= g.min()
    ys, xs = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    cen = np.array([(g * xs).sum() / g.sum(), (g * ys).sum() / g.sum()])
    assert np.linalg.norm(cen - project(A, np.array([[cx, cy]]))[0][0]) < 0.05


def test_g2_warp_direction_independent(texture):
    """Независимая проверка Г2-99: warp(u) = I(π(H⁻¹u)) и совпадение с cv2."""
    rng = np.random.default_rng(5)
    I = texture
    H = random_h(rng, (320, 240), 0.06)                         # I -> E
    out, valid = warp_with_mask(I, H, (320, 240))
    Hinv = np.linalg.inv(H)
    for u in [(100.0, 80.0), (200.0, 150.0), (160.0, 120.0)]:
        p = project(Hinv, np.array([u]))[0][0]
        x0, y0 = int(np.floor(p[0])), int(np.floor(p[1]))
        fx, fy = p[0] - x0, p[1] - y0
        ref = ((1 - fy) * ((1 - fx) * I[y0, x0] + fx * I[y0, x0 + 1])
               + fy * ((1 - fx) * I[y0 + 1, x0] + fx * I[y0 + 1, x0 + 1]))
        assert valid[int(u[1]), int(u[0])]
        assert np.allclose(out[int(u[1]), int(u[0])], ref, atol=1e-12)
    u8 = np.rint(I * 255).astype(np.uint8)
    cv_fwd = cv2.warpPerspective(u8, H, (320, 240), flags=cv2.INTER_LINEAR).astype(float) / 255
    cv_inv = cv2.warpPerspective(u8, Hinv, (320, 240), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                                 ).astype(float) / 255
    ours, v = warp_with_mask(u8.astype(float) / 255, H, (320, 240))
    inner = v.copy()
    inner[:2], inner[-2:], inner[:, :2], inner[:, -2:] = False, False, False, False
    assert np.abs(cv_fwd - ours)[inner].max() < 2.5 / 255     # фиксированная точность весов cv2
    assert np.abs(cv_inv - ours)[inner].max() < 2.5 / 255
    wrong = cv2.warpPerspective(u8, Hinv, (320, 240), flags=cv2.INTER_LINEAR).astype(float) / 255
    assert np.abs(wrong - ours)[inner].mean() > 0.02           # неверная семантика заметна


# ---------------------------------------------------------------- ПРИЁМ-Г3

def test_g3_outside_I_not_compared_and_null_cells(texture):
    E = texture
    W, H = 320, 240
    # I видит только левую половину E: H: I -> E — сдвиг.
    Hm = np.array([[1, 0, -W / 2], [0, 1, 0], [0, 0, 1.0]])
    I, _ = warp_with_mask(E, np.linalg.inv(Hm), (W, H))
    aligned, valid = warp_with_mask(I, Hm, (W, H))
    assert not valid[:, W // 2 + 2:].any()                     # за границей I пиксели не наблюдаемы
    cfg = Config()
    reg = compare_regions(E, aligned, valid & object_mask((W, H)), cfg.compare)
    right = [k for k in range(64) if k % 8 >= 5]
    assert np.isnan(reg.d_mean[right]).all() and np.isnan(reg.a[:, right]).all()
    assert not reg.kval[right].any()


def test_g3_critical_region_fully_hidden_rejects(texture):
    E = texture
    W, H = 320, 240
    vis = np.ones((H, W), bool)
    vis[H // 2 - 40:H // 2 + 40, W // 2 - 50:W // 2 + 50] = False   # закрыт центр
    cfg = Config().replace(compare={"critical": "center4"})
    r = run_pair(E, E.copy(), cfg, "crit", vis_mask_I=vis, alignment="oracle", H_oracle=np.eye(3))
    assert r.q_cmp == 0 and r.y_pred is None
    assert "REQUIRED_REGION_INVISIBLE" in r.decision.reasons


def test_g3_gt_files_do_not_change_working_predictions(tmp_path, texture):
    rng = np.random.default_rng(1)
    E = texture
    Hgt = random_h(rng, (320, 240), 0.03)
    I, _ = warp_with_mask(E, np.linalg.inv(Hgt), (320, 240))
    write_png(tmp_path / "E.png", E)
    write_png(tmp_path / "I.png", I)
    base = {"pair_id": "p", "E_path": str(tmp_path / "E.png"), "I_path": str(tmp_path / "I.png"),
            "y_match": 1, "category": "c"}
    a = pd.DataFrame([{**base, "H_gt": json.dumps(Hgt.tolist())}])
    b = pd.DataFrame([{**base, "H_gt": json.dumps((Hgt * 7 + 1).tolist())}])
    pa, _ = run_inspection(a, tmp_path / "a", Config())
    pb, _ = run_inspection(b, tmp_path / "b", Config())
    cols = ["q_geom", "q_align", "q_cmp", "y_pred", "D_glob", "B_loc", "n_in"]
    pd.testing.assert_frame_equal(pa[cols], pb[cols])


# ---------------------------------------------------------------- ПРИЁМ-Г4

def test_g4_mean_delta_vs_delta_of_means():
    E = np.zeros((64, 64, 3))
    E[:, :32] = [1, 0, 0]
    E[:, 32:] = [0, 1, 0]
    J = E[:, ::-1].copy()                                      # те же цвета, переставлены
    cfg = Config().replace(compare={"grid": 1, "v_min": 0.5})
    reg = compare_regions(E, J, np.ones((64, 64), bool), cfg.compare)
    assert reg.d_mean[0] > 50 and reg.d_of_means[0] < 1e-6


def test_g4_identical_pair_zero_and_fields_match_formulas(texture):
    E = texture
    cfg = Config()
    reg = compare_regions(E, E.copy(), np.ones((240, 320), bool), cfg.compare)
    assert np.allclose(reg.d_mean, 0) and np.allclose(reg.a, 0)
    rng = np.random.default_rng(0)
    J = np.clip(E + rng.normal(0, 0.03, E.shape), 0, 1)
    omega = np.ones((240, 320), bool)
    omega[:, :40] = False
    reg = compare_regions(E, J, omega, cfg.compare, tau_pix=(2.0, 5.0))
    from p2geom.regions import cell_index_map
    cells = cell_index_map((320, 240), 8)
    for k in np.nonzero(reg.kval)[0][:10]:
        m = (cells == k) & omega
        vals = reg.delta_map[m]
        assert np.isclose(reg.d_mean[k], vals.mean(), rtol=0, atol=1e-12)
        assert np.isclose(reg.a[1, k], (vals > 5.0).mean(), atol=1e-12)
        assert np.isclose(reg.v[k], m.sum() / (cells == k).sum())
    dec = decide(1, reg, cfg.decision, cfg.compare)
    w = reg.w[reg.kval]
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(dec.D, np.sum(w * reg.d_mean[reg.kval]))


# ---------------------------------------------------------------- ПРИЁМ-Г5

def test_g5_bad_geometry_and_texture_pairs_stay_in_report(tmp_path, texture):
    rng = np.random.default_rng(2)
    noise1, noise2 = rng.random((240, 320, 3)), rng.random((240, 320, 3))
    write_png(tmp_path / "n1.png", noise1)
    write_png(tmp_path / "n2.png", noise2)
    write_png(tmp_path / "t.png", texture)
    pairs = pd.DataFrame([
        {"pair_id": "noise", "E_path": str(tmp_path / "n1.png"), "I_path": str(tmp_path / "n2.png"),
         "y_match": 1, "category": "c"},
        {"pair_id": "flat", "E_path": str(tmp_path / "t.png"), "I_path": str(tmp_path / "n1.png"),
         "y_match": 0, "category": "c"},
    ])
    pred, _ = run_inspection(pairs, tmp_path / "out", Config())
    assert list(pred["pair_id"]) == ["noise", "flat"]
    assert pred["y_pred"].isna().all()
    assert (pred["reject_reason"] != "").all()
    rc = pd.read_csv(tmp_path / "out" / "reject_counts.csv")
    assert rc["count"].sum() >= 2
