"""P3-E1..E7 experiment configs (spec 10, 8.4): committed files are generated; plans expand via experiments make_plan."""
import copy
import math
from pathlib import Path

import pytest
import yaml

import experiments.cli  # noqa: F401  (registers the experiments params schema)
from experiments.plan import make_plan
from expcore.config import load_config
from expcore.errors import ConfigError
from metric.expconfigs import SELECTION_FREE, materialize
from metric.params import validate_profile

pytestmark = pytest.mark.unit

COMMITTED = Path(__file__).resolve().parents[3] / "configs" / "metric"
E1 = {"selected_params": {"lambda": 0.1},
      "candidates": [{"params": {"lambda": l}, "mean_val_recall_at_1": r}
                     for l, r in ((0, 0.5), (0.01, 0.4), (0.1, 0.45), (1, 0.45))]}
SEL = {"e1_tune": E1, "n_cohort_images": 1000,
       "e4_contrastive": {"selected_params": {"lr": 1e-4, "margin": 0.5}},
       "e4_triplet": {"selected_params": {"lr": 1e-5, "margin": 0.2}},
       "e4_classifier": {"selected_params": {"lr": 3e-4}}}


def _plan(path):
    cfg = load_config(path, module="experiments")
    runs = make_plan(cfg, path.parent)
    for entry in cfg.params.runs:   # every cell passes the profile validator
        validate_profile(load_config(path.parent / entry.config, module="metric"))
    return cfg, runs


def _mat(tmp_path, exp, selected=None):
    return _plan(materialize(exp, tmp_path, selected)[0])


def _cells(path):
    cfg = load_config(path, module="experiments")
    return [yaml.safe_load((path.parent / e.config).read_text(encoding="utf-8")) for e in cfg.params.runs]


@pytest.mark.parametrize("exp", SELECTION_FREE)
def test_committed_files_are_generated(tmp_path, exp):
    for p in materialize(exp, tmp_path):
        assert p.read_bytes() == (COMMITTED / p.relative_to(tmp_path)).read_bytes()


@pytest.mark.parametrize("exp,n_runs,profile", [
    ("smoke", 1, "smoke"), ("e1_tune", 12, "pilot"), ("e4_contrastive", 12, "pilot"), ("e4_triplet", 12, "pilot"),
    ("e4_classifier", 12, "pilot"), ("e4_fixed", 2, "confirmatory"), ("base_minimal", 1, "pilot"),
    ("e6_nol2", 3, "pilot")])
def test_selection_free_plans(exp, n_runs, profile):
    cfg, runs = _plan(COMMITTED / f"{exp}.yaml")
    assert len(runs) == n_runs and cfg.profile == profile


def test_e1_tune_grid():
    cells = _cells(COMMITTED / "e1_tune.yaml")
    assert [c["params"]["calibration"]["lambda"] for c in cells] == [0, 0.01, 0.1, 1]
    assert all(c["params"]["calibration"]["K"] == 32 and c["params"]["calibration"]["n_warm"] == 4 for c in cells)


def test_e4_grids():
    g = lambda e: {(c["params"]["optimizer"]["lr"], c["params"]["loss"]["margin"]) for c in _cells(COMMITTED / f"{e}.yaml")}
    assert g("e4_contrastive") == {(l, m) for l in (1e-5, 1e-4) for m in (0.5, 1)}
    assert g("e4_triplet") == {(l, m) for l in (1e-5, 1e-4) for m in (0.1, 0.2)}
    assert g("e4_classifier") == {(l, None) for l in (1e-5, 1e-4, 3e-4, 1e-3)}


def test_fixed_and_base_minimal():
    fixed = _cells(COMMITTED / "e4_fixed.yaml")
    assert [list(c["algorithms"]) for c in fixed] == [["imagenet_fixed"], ["rgb_hist"]]
    (bm,) = _cells(COMMITTED / "base_minimal.yaml")
    assert bm["params"]["calibration"]["queues"] is False and bm["params"]["calibration"]["lambda"] == 0
    assert bm["seeds"]["seeds"] == [11]


def test_no_tuning_config_is_confirmatory():
    for exp in ("e1_tune", "e4_contrastive", "e4_triplet", "e4_classifier", "base_minimal", "e6_nol2"):
        assert yaml.safe_load((COMMITTED / f"{exp}.yaml").read_text(encoding="utf-8"))["profile"] == "pilot"


def test_e1_confirm(tmp_path):
    cfg, runs = _mat(tmp_path, "e1_confirm", SEL)
    assert cfg.profile == "confirmatory" and len(runs) == 2 * 5
    zero = copy.deepcopy(SEL) | {"e1_tune": {**E1, "selected_params": {"lambda": 0}}}
    assert len(_mat(tmp_path / "z", "e1_confirm", zero)[1]) == 5   # lambda*=0: one distinct cell


def test_e2_e3_use_best_positive_lambda(tmp_path):
    cells = _cells(materialize("e2_k", tmp_path, SEL)[0])
    assert [c["params"]["calibration"]["K"] for c in cells] == [8, 32, 128]
    assert len(_mat(tmp_path / "a", "e2_k", SEL)[1]) == 9
    zero = SEL | {"e1_tune": {**E1, "selected_params": {"lambda": 0}}}
    cells = _cells(materialize("e3_nwarm", tmp_path / "b", zero)[0])
    assert [c["params"]["calibration"]["n_warm"] for c in cells] == [1, 4, 8]
    assert {c["params"]["calibration"]["lambda"] for c in cells} == {0.1}   # M-D20: best lambda>0 (tie -> smaller)
    assert len(_mat(tmp_path / "c", "e3_nwarm", SEL)[1]) == 9


def test_e5_limited(tmp_path):
    cfg, runs = _mat(tmp_path, "e5_limited", SEL)
    assert len(runs) == 8 * 5 and cfg.profile == "confirmatory"
    cells = _cells(tmp_path / "e5_limited.yaml")
    assert {c["params"]["data"]["per_class_limit"] for c in cells} == {2, 3, 4, None}
    assert {c["params"]["data"]["cohort_min_images"] for c in cells} == {4}
    assert {c["params"]["train"]["steps_per_epoch"] for c in cells} == {math.ceil(1000 / 32)}
    assert {c["params"]["calibration"]["lambda"] for c in cells} == {0.1, 0}
    with pytest.raises(ConfigError):
        materialize("e5_limited", tmp_path / "n", {"e1_tune": E1})


def test_e6_noaug_and_e7(tmp_path):
    cfg, runs = _mat(tmp_path, "e6_noaug", SEL)
    assert len(runs) == 3 and cfg.profile == "pilot"
    assert _cells(tmp_path / "e6_noaug.yaml")[0]["params"]["input"]["train_augment"] is False
    cfg, runs = _mat(tmp_path / "e7", "e7_degrade", SEL)
    cells = _cells(tmp_path / "e7" / "e7_degrade.yaml")
    assert {c["params"]["degradation"] for c in cells} == {"bright_+0.1", "bright_-0.1", "noise_0.03"}
    assert cfg.profile == "confirmatory" and {e.command for e in cfg.params.runs} == {"search"}
    # models: E1 lambda*, lambda=0, 3 selected E4, imagenet_fixed, rgb_hist
    assert len(cells) == 3 * 7 and len(runs) == 3 * 7 * 5


def test_e4_confirm(tmp_path):
    cfg, runs = _mat(tmp_path, "e4_confirm", SEL)
    assert cfg.profile == "confirmatory" and len(runs) == 3 * 5
    assert {e.command for e in cfg.params.runs} == {"train"}
    for exp in ("e4_confirm", "e7_degrade"):
        with pytest.raises(ConfigError):
            materialize(exp, tmp_path / exp, {k: v for k, v in SEL.items() if k != "e4_triplet"})


@pytest.mark.parametrize("exp", ["e4_confirm", "e1_confirm", "e2_k", "e3_nwarm", "e5_limited", "e6_noaug", "e7_degrade"])
def test_selection_required(tmp_path, exp):
    with pytest.raises(ConfigError):
        materialize(exp, tmp_path)
