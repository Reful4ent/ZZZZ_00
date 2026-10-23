import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from expcore.config import config_hash, load_config, register_params_schema, strict_from_dict
from expcore.errors import ConfigError

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "templates" / "full.yaml"


@dataclass(frozen=True)
class TemplateParams:
    pass


register_params_schema("template", TemplateParams)


def _load(tmp_path, d, module="template", **kw):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return load_config(p, module=module, **kw)


@pytest.fixture
def full_cfg_dict():
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture
def assert_config_error(tmp_path):
    def check(d, module="template", match=None):
        with pytest.raises(ConfigError, match=match):
            _load(tmp_path, d, module)
    return check


def _as_smoke(d):
    d = copy.deepcopy(d)
    d["profile"] = "smoke"
    d["seeds"] = {"seed": 11, "seeds": [11]}
    for ds in d["datasets"].values():
        ds["fixture"] = True
    d["datasets"]["kadid10k"]["n_validation"] = 2
    return d


def test_full_template_loads(tmp_path, full_cfg_dict):
    cfg = _load(tmp_path, full_cfg_dict)
    assert cfg.profile == "pilot" and cfg.datasets["kadid10k"].n_validation == 20
    assert cfg.inputs.pair_columns == ("pair_id", "query_id", "reference_id", "relation_id")
    assert cfg.runner.chunk_size == 256 and cfg.params == TemplateParams()


def test_smoke_variant_loads(tmp_path, full_cfg_dict):
    assert _load(tmp_path, _as_smoke(full_cfg_dict)).profile == "smoke"


def test_missing_key_raises(full_cfg_dict, assert_config_error):
    del full_cfg_dict["decision"]["tie_rule"]
    assert_config_error(full_cfg_dict, match="decision.tie_rule")


def test_unknown_key_raises(full_cfg_dict, assert_config_error):
    full_cfg_dict["extra"] = 1
    assert_config_error(full_cfg_dict, match="extra")


def test_wrong_type_raises(full_cfg_dict, assert_config_error):
    full_cfg_dict["runner"]["chunk_size"] = "256"
    assert_config_error(full_cfg_dict, match="runner.chunk_size")


def test_paths_excluded_from_hash(tmp_path, full_cfg_dict):
    other = copy.deepcopy(full_cfg_dict)
    other["paths"]["data_root"] = "/elsewhere/data"
    a = _load(tmp_path, full_cfg_dict)
    b = _load(tmp_path, other)
    assert a.paths != b.paths
    assert config_hash(a) == config_hash(b)
    full_cfg_dict["runner"]["chunk_size"] = 128
    assert config_hash(_load(tmp_path, full_cfg_dict)) != config_hash(a)


def test_forbidden_pair_columns(full_cfg_dict, assert_config_error):
    for col in ("y_match", "gt_H_path", "gt_mask_path"):
        full_cfg_dict["inputs"]["pair_columns"] = ["pair_id", "query_id", "reference_id", "relation_id", col]
        assert_config_error(full_cfg_dict, match=col)


def test_seed_must_match_profile(tmp_path, full_cfg_dict, assert_config_error):
    full_cfg_dict["seeds"]["seed"] = 23
    assert _load(tmp_path, full_cfg_dict).seeds.seed == 23
    full_cfg_dict["seeds"]["seed"] = 51
    assert_config_error(full_cfg_dict, match="seed")


def test_fixture_dataset_only_in_smoke(full_cfg_dict, assert_config_error):
    full_cfg_dict["datasets"]["sop"]["fixture"] = True
    assert_config_error(full_cfg_dict, match="fixture")


def test_unregistered_params_schema(full_cfg_dict, assert_config_error):
    assert_config_error(full_cfg_dict, module="unknown_module", match="unknown_module")


def test_real_kadid_requires_n_validation_20(full_cfg_dict, assert_config_error):
    full_cfg_dict["datasets"]["kadid10k"]["n_validation"] = 3
    assert_config_error(full_cfg_dict, match="n_validation")


def test_path_override_does_not_change_hash(tmp_path, full_cfg_dict):
    a = _load(tmp_path, full_cfg_dict)
    b = _load(tmp_path, full_cfg_dict, path_overrides={"out_root": tmp_path / "o"})
    assert b.paths.out_root == tmp_path / "o" and b.paths.data_root == a.paths.data_root
    assert config_hash(a) == config_hash(b)


def test_unknown_path_override_raises(tmp_path, full_cfg_dict):
    with pytest.raises(ConfigError, match="bogus"):
        _load(tmp_path, full_cfg_dict, path_overrides={"bogus": tmp_path})


def test_strict_from_dict_nested():
    @dataclass(frozen=True)
    class Inner:
        x: int

    @dataclass(frozen=True)
    class Outer:
        inner: Inner
        tags: tuple[str, ...]
        note: str | None

    assert strict_from_dict(Outer, {"inner": {"x": 1}, "tags": ["a"], "note": None}) == Outer(Inner(1), ("a",), None)
    with pytest.raises(ConfigError, match="inner.y"):
        strict_from_dict(Outer, {"inner": {"x": 1, "y": 2}, "tags": [], "note": None})


def test_duplicate_yaml_key_raises(tmp_path):   # review: last duplicate silently won
    text = TEMPLATE.read_text(encoding="utf-8").replace('tie_rule: ">="', 'tie_rule: ">"\n  tie_rule: ">="')
    p = tmp_path / "dup.yaml"; p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="tie_rule"):
        load_config(p, module="template")
