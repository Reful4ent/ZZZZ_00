import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from expcore.config import config_hash, load_config
from expcore.errors import ConfigError
from metric.params import BASE_PARAMS, MetricParams, params_dict, validate_profile

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "configs" / "templates" / "metric.yaml"
SMOKE = REPO / "configs" / "smoke" / "metric.yaml"


def _cfg(tmp_path, profile="pilot", algorithms=None, src=TEMPLATE, **sections):
    """Load a variant of the template; sections like `batch={"classes": 8}` merge into params sections."""
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    d["profile"] = profile
    if profile == "confirmatory":
        d["seeds"] = {"seed": 11, "seeds": [11, 23, 37, 51, 71]}
    if algorithms is not None:
        d["algorithms"] = algorithms
    for sec, upd in sections.items():
        d["params"][sec].update(upd)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return load_config(p, module="metric")


def test_template_valid_all_profiles(tmp_path):
    for profile in ("pilot", "confirmatory"):
        validate_profile(_cfg(tmp_path, profile))
    validate_profile(load_config(SMOKE, module="metric"))


def test_template_matches_base_params():
    assert yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))["params"] == BASE_PARAMS


def test_loads_via_expcore():
    cfg = load_config(TEMPLATE, module="metric")
    assert isinstance(cfg.params, MetricParams)
    d = params_dict(cfg.params)
    assert d["calibration"]["lambda"] == 0.0 and d["train"]["steps_per_epoch"] == "auto"
    assert list(d["search"]["k"]) == [1, 5, 10]


@pytest.mark.parametrize("profile", ["pilot", "confirmatory"])
def test_rejects_batch_change(tmp_path, profile):
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, profile, batch={"classes": 8}))


def test_rejects_lr_change_for_proxy(tmp_path):
    opt = {"lr": 3e-4}
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, "confirmatory", optimizer=opt))
    validate_profile(_cfg(tmp_path, "confirmatory", {"classifier": "softmax_v1"}, optimizer=opt))


def test_rejects_subset_confirmatory(tmp_path):
    validate_profile(_cfg(tmp_path, "pilot", data={"class_subset": 5}))
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, "confirmatory", data={"class_subset": 5}))


def test_rejects_nwarm_gt_K(tmp_path):
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, calibration={"n_warm": 40, "K": 32}))


def test_unknown_key(tmp_path):
    with pytest.raises(ConfigError):
        _cfg(tmp_path, batch={"extra": 1})


def test_loss_id_in_hash(tmp_path):
    a = _cfg(tmp_path, algorithms={"proxy_cal": "proxy_cos_softmax_v1"})
    b = _cfg(tmp_path, algorithms={"proxy_cal": "proxy_cos_softmax_v2"})
    assert config_hash(a) != config_hash(b)


def test_algorithms_rules(tmp_path):
    for algs in ({}, {"proxy_cal": "v", "triplet": "v"}, {"nope": "v"}):
        with pytest.raises(ConfigError):
            validate_profile(_cfg(tmp_path, algorithms=algs))


def test_queues_false_only_for_non_proxy_cal(tmp_path):
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, calibration={"queues": False}))
    validate_profile(_cfg(tmp_path, algorithms={"base_minimal": "v"}, calibration={"queues": False}))


def test_real_weights_required_outside_smoke(tmp_path):
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, model={"weights": None}))
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, model={"weights_sha256": "0" * 64}))


def test_smoke_allows_only_md27_substitutions(tmp_path):
    validate_profile(_cfg(tmp_path, "smoke", src=SMOKE, batch={}))
    with pytest.raises(ConfigError):
        validate_profile(_cfg(tmp_path, "smoke", src=SMOKE, batch={"classes": 8}))
