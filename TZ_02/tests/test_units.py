"""Модульные тесты компонентов."""
import numpy as np
import pandas as pd
import pytest

from conftest import random_h
from p2geom.calibration import RegionStack, calibrate_color, calibrate_geometry, recompute
from p2geom.config import Config, DecisionConfig, GeometryGateConfig, RansacConfig, load_config, save_config
from p2geom.datasets.hpatches import check_direction
from p2geom.features import Features
from p2geom.homography import check_admissible, hull_coverage, project
from p2geom.io import correspondence_frame, prediction_record, region_frame
from p2geom.matching import match_descriptors
from p2geom.ransac import estimate_homography, refine_homography
from p2geom.synth import apply_defect, factor_cases, generate_case, occluder, perspective_matrix
from p2geom.warp import warp_with_mask


def _feat(desc):
    n = len(desc)
    return Features("sift", "OK", np.zeros((n, 2)), np.zeros((n, 2)), np.asarray(desc, np.float32), "l2",
                    np.eye(3), (10, 10), (10, 10))


def test_matching_ratio_then_mutual_and_ties():
    dI = [[0, 0], [10, 0], [0, 10], [5, 5]]
    dE = [[0, 0.1], [10, 0.1], [0, 10.1], [5, 5.1], [5, 5.1]]   # E3 и E4 равны -> d2 = d1 -> ratio 1
    m = match_descriptors(_feat(dI), _feat(dE), 0.75)
    assert {tuple(p) for p in m.pairs} == {(0, 0), (1, 1), (2, 2)}
    assert m.counts["mutual"] == 3
    assert len(set(m.pairs[:, 1])) == len(m.pairs)
    assert len(match_descriptors(_feat(dI[:1]), _feat(dE), 0.75)) == 0   # меньше двух соседей


def test_ransac_outliers_and_trial_log():
    rng = np.random.default_rng(0)
    Ht = random_h(rng, (640, 480), 0.05)
    x = rng.uniform([0, 0], [640, 480], (300, 2))
    u = project(Ht, x)[0] + rng.normal(0, 0.5, (300, 2))
    out = rng.random(300) < 0.4
    u[out] = rng.uniform([0, 0], [640, 480], (out.sum(), 2))
    cfg = RansacConfig(adaptive=False, max_trials=500)
    est = estimate_homography(x, u, cfg)
    assert est.n_trials == 500 and len(est.trial_log) == 500
    assert est.stop_reason == "max_trials"
    assert est.inliers[~out].mean() > 0.95 and est.inliers[out].mean() < 0.05
    est2 = estimate_homography(x, u, cfg)
    assert np.array_equal(est.H, est2.H)                          # детерминизм по seed
    est3 = estimate_homography(x, u, RansacConfig())
    assert est3.stop_reason == "adaptive" and est3.n_trials <= est3.required_trials


def test_refine_failed_keeps_h0():
    rng = np.random.default_rng(1)
    Ht = random_h(rng, (400, 300), 0.05)
    x = rng.uniform([0, 0], [400, 300], (50, 2))
    u = project(Ht, x)[0] + rng.normal(0, 0.3, (50, 2))
    H0 = Ht.copy()
    H0[0, 2] += 0.5
    H, status, sse0, _ = refine_homography(H0, x, u, 100, lambda H: False)
    assert status == "REFINE_FAILED"
    assert np.allclose(H, H0 / H0.flat[np.argmax(np.abs(H0))])
    H, status, sse0, sse1 = refine_homography(H0, x, u, 100, lambda H: True)
    assert status == "OK" and sse1 <= sse0


def test_admissibility_rules():
    size = (100, 80)
    assert check_admissible(np.eye(3), size, size).ok
    assert not check_admissible(np.diag([1, 1, 0.0]), size, size).ok
    assert "JACOBIAN_SIGN" in check_admissible(np.diag([-1, 1, 1.0]), size, size).reasons   # зеркало
    Hp = np.eye(3)
    Hp[2, 0] = -1.5 / 99                                         # линия горизонта внутри ROI
    assert "DENOMINATOR_SIGN_CHANGE" in check_admissible(Hp, size, size).reasons
    assert "SCALE_OUT_OF_RANGE" in check_admissible(np.diag([3, 3, 1.0]), size, size,
                                                    area_ratio_range=(0.5, 2.0)).reasons
    assert check_admissible(None, size, size).reasons == ["NO_MODEL"]


def test_hull_coverage_clipped():
    pts = np.array([[-50, -50], [49.5, -50], [49.5, 39.5], [-50, 39.5]])
    assert np.isclose(hull_coverage(pts, (100, 80)), 0.25)
    assert hull_coverage(pts[:2], (100, 80)) == 0.0


def test_direction_check(texture):
    rng = np.random.default_rng(4)
    H = random_h(rng, (320, 240), 0.05)
    I, _ = warp_with_mask(texture, H, (320, 240))
    assert check_direction(texture, I, H)["verdict"] == "E_TO_I"
    assert check_direction(texture, I, np.linalg.inv(H))["verdict"] == "I_TO_E"


def test_generator_cases_and_occluder():
    cases = factor_cases()
    assert sum(c["factor"] == "perspective" for c in cases) == 4 * 5
    assert sum(c["factor"] == "combined" for c in cases) == 8
    rng = np.random.default_rng(0)
    m, meta = occluder((200, 100), 0.25, rng)
    assert abs(meta["occ_area_px"] / 20000 - 0.25) < 0.02 and m.sum() == meta["occ_area_px"]
    G, att = perspective_matrix(0.10, (200, 100), np.random.default_rng(1))
    assert att >= 1 and np.isfinite(G).all()


def test_generate_case_records_meta(texture):
    case = next(c for c in factor_cases() if c["factor"] == "rotation" and c["value"] == 30.0)
    obs = generate_case(texture, case)
    assert 0.5 < obs.visibility < 1.0
    assert np.allclose(obs.H_gt @ obs.G, np.eye(3))
    assert {"seed", "attempts", "G", "visibility"} <= set(obs.meta)


@pytest.mark.parametrize("spec", [{"type": "color", "area": 0.02, "channel": "a", "shift": 10.0},
                                  {"type": "scratch", "area": 0.02},
                                  {"type": "paste", "area": 0.02}])
def test_defects_area_and_lightness(texture, spec):
    rng = np.random.default_rng(0)
    out, mask, info = apply_defect(texture, spec, rng, donor=texture[::-1, ::-1].copy())
    assert abs(info["actual_area_frac"] - 0.02) < 0.004
    assert not (out[~mask] != texture[~mask]).any()
    assert info["mean_dE00"] > 0
    if spec["type"] == "color":
        from p2geom.color import srgb_to_lab
        dl = np.abs(srgb_to_lab(out[mask])[:, 0] - srgb_to_lab(texture[mask])[:, 0])
        assert np.median(dl) < 1.0                                # L* сохраняется (до clip)


def test_geometry_calibration_tie_break():
    rows = []
    for tau in (0.65, 0.75, 0.85):
        for k in range(10):
            rows.append({"ratio": tau, "negative": False, "n_in": 50, "rho_in": 0.9, "med_err": 0.5,
                         "c_I": 0.5, "c_E": 0.5, "adm_ok": True, "e_H": 1.0})
            rows.append({"ratio": tau, "negative": True, "n_in": 6, "rho_in": 0.5, "med_err": 1.0,
                         "c_I": 0.02, "c_E": 0.02, "adm_ok": True, "e_H": None})
    table, best = calibrate_geometry(pd.DataFrame(rows), GeometryGateConfig())
    assert len(table) == 3 * 16
    # Все положительные проходят при любых порогах; отрицательные отсекаются n_min>=8 или c_min>=0.05.
    assert best["pos_success"] == 1.0 and best["neg_accept"] == 0.0
    assert best["n_min"] == 20 and best["c_min"] == 0.20 and best["ratio"] == 0.65


def test_color_calibration_separable_and_infeasible():
    n, K = 40, 4
    y = np.array([1] * 20 + [0] * 20)
    d = np.where(y[:, None] == 1, 1.0, 8.0) * np.ones((n, K))
    st = RegionStack([str(i) for i in range(n)], np.ones((n, K), bool), d, d, np.full((n, K), 0.25),
                     {t: np.zeros((n, K)) for t in (2.0, 5.0, 10.0)})
    table, best = calibrate_color(y, np.ones(n, bool), st, [], "mean_delta", DecisionConfig())
    assert best["status"] == "ok" and best["recall_all"] == 1.0 and best["fpr_all"] == 0.0
    _, best = calibrate_color(y, np.zeros(n, bool), st, [], "mean_delta", DecisionConfig())
    assert best["status"] == "infeasible"


def test_recompute_matches_pipeline(texture):
    """Пересчёт по таблицам (абляции) совпадает с решением конвейера."""
    from p2geom.pipeline import run_pair
    rng = np.random.default_rng(3)
    Hgt = random_h(rng, (320, 240), 0.03)
    I, _ = warp_with_mask(texture, np.linalg.inv(Hgt), (320, 240))
    I2 = I.copy()
    I2[100:140, 140:200] = [0.9, 0.1, 0.1]
    cfgs = [Config(), Config().replace(compare={"critical": "center4"}, decision={"tau_D": 1.0}),
            Config().replace(compare={"region_stat": "delta_of_means"}, decision={"use_B": False})]
    for cfg in cfgs:
        recs, regs = [], []
        for pid, img in (("a", I), ("b", I2)):
            r = run_pair(texture, img, cfg, pid)
            recs.append(prediction_record(r))
            regs.append(region_frame(r))
            assert len(correspondence_frame(r)) == r.geom.n0
        pred = pd.DataFrame(recs)
        rc = recompute(pred, pd.concat(regs, ignore_index=True), cfg)
        assert list(rc["q_cmp"]) == list(pred["q_cmp"])
        assert np.allclose(rc["D_glob"].astype(float), pred["D_glob"].astype(float))
        assert list(rc["y_pred"].fillna(-1)) == list(pred["y_pred"].fillna(-1))


def test_config_roundtrip(tmp_path):
    cfg = Config().replace(gate={"n_min": 8, "area_ratio_range": (0.2, 3.0)})
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
