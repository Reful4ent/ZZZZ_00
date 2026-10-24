import pytest

from expcore import registry
from expcore.config import config_hash, load_config
from expcore.io import load_json
from expcore.protocol import open_test
from experiments.cli import main
from tests.cli_helpers import make_world
from tests.fakes import RandomScorePredictor

pytestmark = pytest.mark.unit


def _plan_dir(world, profile):
    return world.out_root / profile / "plan" / "1"


@pytest.mark.parametrize("profile,seeds", [("pilot", {11, 23, 37}), ("confirmatory", {11, 23, 37, 51, 71})])
def test_plan_seeds_by_profile(tmp_path, monkeypatch, profile, seeds):
    world = make_world(tmp_path, monkeypatch, profile=profile)
    assert main(world.exp_argv("plan")) == 0
    plan = load_json(_plan_dir(world, profile) / "plan.json")
    assert {r["seed"] for r in plan["runs"]} == seeds and {r["module"] for r in plan["runs"]} == {"photometric"}


def test_plan_lock_written_for_confirmatory_only(tmp_path, monkeypatch):
    pilot = make_world(tmp_path / "p", monkeypatch, profile="pilot")
    assert main(pilot.exp_argv("plan")) == 0
    assert not (_plan_dir(pilot, "pilot") / "plan.lock.json").exists()
    conf = make_world(tmp_path / "c", monkeypatch, profile="confirmatory")
    assert main(conf.exp_argv("plan")) == 0
    lock = load_json(_plan_dir(conf, "confirmatory") / "plan.lock.json")
    plan = load_json(_plan_dir(conf, "confirmatory") / "plan.json")
    assert sorted(lock["config_hashes"]) == sorted(r["config_hash"] for r in plan["runs"])


def test_run_ids_match_plan(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    assert main(world.exp_argv("plan")) == 0
    pdir = _plan_dir(world, "pilot")
    for r in load_json(pdir / "plan.json")["runs"]:
        cfg = load_config(pdir / r["config_path"], module="photometric")
        assert cfg.seeds.seed == r["seed"] and config_hash(cfg) == r["config_hash"]
        assert r["run_id"] == f"photometric-pilot-{r['config_hash'][:12]}-s{r['seed']}"


def test_plan_after_test_open_exits_5(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="confirmatory")
    assert main(world.exp_argv("plan")) == 0
    pdir = _plan_dir(world, "confirmatory")
    open_test(world.registry_path, pdir, "1", load_json(pdir / "plan.json")["runs"][0]["run_id"])
    assert main(world.exp_argv("plan")) == 0   # unchanged plan: allowed
    world.cfg_dict["runner"]["chunk_size"] = 7   # a changed module config is a changed plan
    world.write()
    assert main(world.exp_argv("plan")) == 5


def test_run_executes_plan_with_fake_algorithm(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    assert main(world.exp_argv("plan")) == 0
    assert main(world.exp_argv("run")) == 0
    for r in load_json(_plan_dir(world, "pilot") / "plan.json")["runs"]:
        assert (world.out_root / "pilot" / "photometric" / r["run_id"] / "_SUCCESS").is_file()


def test_run_without_algorithm_returns_6(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    assert main(world.exp_argv("plan")) == 0
    assert main(world.exp_argv("run")) == 6


def test_confirmatory_run_outside_plan_exits_5(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="confirmatory")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    assert main(world.exp_argv("plan")) == 0
    pdir = _plan_dir(world, "confirmatory")
    edited = pdir / load_json(pdir / "plan.json")["runs"][0]["config_path"]
    edited.write_text(edited.read_text(encoding="utf-8").replace("chunk_size: 256", "chunk_size: 128"),
                      encoding="utf-8")   # changed after planning: its hash is not in plan.lock.json
    assert main(world.exp_argv("run")) == 5


def test_run_without_plan_exits_3(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    assert main(world.exp_argv("run")) == 3


def test_module_profile_must_match(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    world.exp_dict["profile"] = "confirmatory"
    world.exp_dict["seeds"] = {"seed": 11, "seeds": [11]}
    from tests.conftest import write_cfg
    write_cfg(world.exp_dict, world.exp_path)
    assert main(world.exp_argv("plan")) == 2
