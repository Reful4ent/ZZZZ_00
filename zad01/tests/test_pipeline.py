"""Дымовой прогон всего конвейера на фиктивной базе в формате KADID-10k."""

import json

import pandas as pd
import pytest
import yaml

from photometric.cli import main
from photometric.config import load_config
from photometric.kadid import load_kadid
from photometric.methods import DEFAULT_METHOD_IDS

from conftest import make_fake_kadid


def _write_cfg(tmp_path, kadid_root, out, **over):
    cfg = {
        "output_dir": str(out),
        "kadid": {"root": str(kadid_root), "validate_full": False},
        "experiments": {
            "P1-E2": {"noise_seeds": 2, "brightness": [-0.1, 0.0, 0.1], "contrast": [1.0, 1.4], "gamma": [1.0, 1.6],
                      "noise_sigma": [0.0, 0.05], "jpeg_quality": [90, 30]},
            "P1-E4": {"n_images": 2, "warmup": 1, "repeats": 2},
        },
        "split": {"fractions": {"validation": 0.5, "test": 0.5}},
    }
    for k, v in over.items():
        cfg[k] = v
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    root = make_fake_kadid(tmp / "kadid", n_originals=10, h=24, w=32)
    out = tmp / "out"
    cfgp = _write_cfg(tmp, root, out)
    assert main(["run", "--config", str(cfgp)]) == 0
    return tmp, root, out, cfgp


def test_outputs_exist(run_dir):
    _, _, out, _ = run_dir
    for f in ["predictions.parquet", "thresholds.json", "manifest.parquet", "run_meta.json", "split.csv",
              "metrics/e1_overall.csv", "metrics/e1_by_type_level.csv", "metrics/e1_macro_original.csv",
              "metrics/e2_curves.csv", "metrics/e3_summary.csv", "metrics/e3_histogram_invariance.csv",
              "metrics/e4_timing.csv", "curves/test/e2_noise_decisions.png", "curves/test/e1_recall_by_level.png"]:
        assert (out / f).is_file(), f
    assert any((out / "representation_cache").rglob("*.npz"))


def test_every_pair_every_method(run_dir):
    _, _, out, _ = run_dir
    pred = pd.read_parquet(out / "predictions.parquet")
    man = pd.read_parquet(out / "manifest.parquet")
    assert len(pred) == len(man) * len(DEFAULT_METHOD_IDS)
    counts = pred.groupby("pair_id")["method_id"].nunique()
    assert (counts == len(DEFAULT_METHOD_IDS)).all()
    # E1: 10 оригиналов × 11 типов × 5 уровней × 4 пары
    assert (man["experiment"] == "P1-E1").sum() == 10 * 11 * 5 * 4
    assert set(pred["decision"].unique()) <= {0, 1}
    assert ((pred["status"] != "OK") <= (pred["decision"] == 0)).all()


def test_thresholds_file(run_dir):
    _, _, out, _ = run_dir
    th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    assert set(th) == set(DEFAULT_METHOD_IDS)
    for m, r in th.items():
        assert r["fpr_all"] <= 0.05 and r["alpha"] == 0.05
        assert r["tuned_on"] == {"experiment": "P1-E1", "split": "validation"}


def test_e3_histogram_blind_to_permutation(run_dir):
    _, _, out, _ = run_dir
    inv = pd.read_csv(out / "metrics" / "e3_histogram_invariance.csv")
    assert inv["hist_equal"].all() and (abs(inv["shi"] - 1.0) <= 1e-12).all() and (inv["pixels_changed_fraction"] > 0.5).all()
    summ = pd.read_csv(out / "metrics" / "e3_summary.csv").set_index("method_id")
    assert summ.loc["hist_rgb8", "permuted_indistinguishable_from_identity"]
    assert summ.loc["hist_rgb8", "raw_permuted_mean"] == pytest.approx(1.0)
    assert summ.loc["mse", "raw_permuted_mean"] > 0
    assert summ.loc["zncc", "raw_permuted_mean"] < 0.5


def test_e2_controls_equal_identity(run_dir):
    _, _, out, _ = run_dir
    pred = pd.read_parquet(out / "predictions.parquet")
    ctrl = pred[(pred["experiment"] == "P1-E2") & (pred["label"] == 1) &
                (((pred["factor"] == "brightness") & (pred["param"] == 0.0)) |
                 ((pred["factor"].isin(["contrast", "gamma"])) & (pred["param"] == 1.0)) |
                 ((pred["factor"] == "noise") & (pred["param"] == 0.0)))]
    assert len(ctrl)
    mse = ctrl[ctrl["method_id"] == "mse"]["raw_value"]
    assert (mse == 0).all()


def test_rerun_same_manifest(run_dir, tmp_path):
    tmp, root, out, _ = run_dir
    out2 = tmp_path / "out2"
    cfgp = _write_cfg(tmp_path, root, out2)
    main(["run", "--config", str(cfgp)])
    m1 = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "run_meta.json").read_text(encoding="utf-8"))
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    p1 = pd.read_parquet(out / "predictions.parquet").sort_values(["pair_id", "method_id"])
    p2 = pd.read_parquet(out2 / "predictions.parquet").sort_values(["pair_id", "method_id"])
    pd.testing.assert_series_equal(p1["raw_value"].reset_index(drop=True), p2["raw_value"].reset_index(drop=True))


def test_full_composition_check(tmp_path):
    root = make_fake_kadid(tmp_path / "k", n_originals=81, h=4, w=4, full_names=True)
    df = load_kadid(root)
    assert (df["type_id"] == "00").sum() == 81 and (df["type_id"] != "00").sum() == 10125
    (root / "images" / "I05_13_02.png").unlink()
    with pytest.raises(ValueError, match="composition"):
        load_kadid(root)


def test_config_rejects_dropping_levels(tmp_path):
    p = _write_cfg(tmp_path, tmp_path, tmp_path / "o")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["kadid"]["levels"] = [1, 2, 3]
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
