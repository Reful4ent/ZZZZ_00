import dataclasses
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from expcore import config, registry
from expcore.config import config_hash, load_config
from expcore.io import load_json
from expcore.protocol import load_registry, write_plan_lock
from expcore.resume import read_predictions
from expcore.runlog import make_run_id
from tests.cli_helpers import make_world
from tests.fakes import CrashOnUnitPredictor, RandomScorePredictor

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class EmptyParams:
    """Stands in for the real metric schema so these tests keep exercising expcore mechanics only."""


@pytest.fixture
def empty_metric_params(monkeypatch):
    monkeypatch.setitem(config._PARAMS_SCHEMAS, "metric", EmptyParams)

REPO = Path(__file__).resolve().parents[2]
MODULE_CMDS = [("photometric", ["run"]), ("geometric", ["run"]), ("metric", ["train"]), ("metric", ["embed"]),
               ("metric", ["search"])]


def _main(module):
    return importlib.import_module(f"{module}.cli").main


def _run_dir(world, module, profile="smoke"):
    cfg = load_config(world.cfg_path, module=module)
    return world.out_root / profile / module / make_run_id(module, profile, config_hash(cfg), cfg.seeds.seed)


@pytest.mark.parametrize("mod,cmd", MODULE_CMDS)
def test_help_exits_zero(mod, cmd):   # T-CLI-01
    assert subprocess.run([sys.executable, "-m", mod, *cmd, "--help"], capture_output=True, cwd=REPO).returncode == 0


@pytest.mark.parametrize("mod,cmd", MODULE_CMDS)
def test_skeleton_returns_6(tmp_path, monkeypatch, mod, cmd, request):
    if mod == "metric":
        request.getfixturevalue("empty_metric_params")
    world = make_world(tmp_path, monkeypatch)
    assert _main(mod)(world.argv(*cmd)) == 6
    run = load_json(_run_dir(world, mod) / "run.json")
    assert run["status"] == "failed" and run["failure"]["type"] == "NotImplementedAlgorithm"
    assert run["command"] == cmd[0]


def test_missing_config_exit_2(tmp_path):
    main = _main("photometric")
    assert main(["run", "--config", str(tmp_path / "nope.yaml")]) == 2
    assert main(["run"]) == 2   # --config is required


def test_two_algorithms_exit_2(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, algorithms={"a": "1", "b": "1"}, build=False)
    assert _main("photometric")(world.argv("run")) == 2


def test_unfrozen_manifest_exit_3(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, build=False)
    assert _main("photometric")(world.argv("run")) == 3


def test_registered_fake_algorithm_runs(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    assert _main("photometric")(world.argv("run")) == 0
    run_dir = _run_dir(world, "photometric")
    assert (run_dir / "_SUCCESS").is_file() and load_json(run_dir / "run.json")["status"] == "completed"
    pred = read_predictions(run_dir)
    assert pred.pair_id.str.startswith("p1:").all() and set(pred.method_id) == {"fake"}
    assert _main("photometric")(world.argv("run")) == 0   # completed run: verified, not recomputed
    assert len(load_json(run_dir / "run.json")["attempts"]) == 1


def test_metric_search_runs_queries(tmp_path, monkeypatch, empty_metric_params):
    from tests.fakes import BatchQueryPredictor
    world = make_world(tmp_path, monkeypatch)
    monkeypatch.setitem(registry._ALGORITHMS, ("metric", "fake"), BatchQueryPredictor())
    assert _main("metric")(world.argv("search")) == 0
    pred = read_predictions(_run_dir(world, "metric"))
    assert pred.pair_id.isna().all() and pred.query_id.is_unique and set(pred.timing_mode) == {"amortized_batch"}


def test_oracle_method_id_suffix(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    world.cfg_dict["data_modes"]["oracle"] = {"enabled": True, "reason": "upper bound"}
    world.write()
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    assert _main("photometric")(world.argv("run")) == 0
    run_dir = _run_dir(world, "photometric")
    assert load_json(run_dir / "run.json")["oracle_mode"] is True
    assert set(read_predictions(run_dir).method_id) == {"fake+oracle"}


def test_metric_import_does_not_import_tensorflow():
    code = "import sys, metric.cli; assert 'tensorflow' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=REPO).returncode == 0


def test_no_absolute_dev_paths_in_source():
    pattern = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")
    hits = [str(p) for p in (REPO / "src").rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert hits == []


def test_technical_error_exit_4(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), CrashOnUnitPredictor(crash_unit=0))
    assert _main("photometric")(world.argv("run")) == 4
    run_dir = _run_dir(world, "photometric")
    assert load_json(run_dir / "run.json")["status"] == "failed"
    assert (run_dir / "errors.jsonl").read_text(encoding="utf-8").strip()
    assert not set(read_predictions(run_dir).get("outcome", [])) - {"decided"}


class _AssertTestOpened(RandomScorePredictor):
    def __init__(self, marker):
        self._marker = marker

    def predict(self, pair, images, ctx):
        assert self._marker.exists(), "test image loaded before test_opened.json"
        return super().predict(pair, images, ctx)


def test_confirmatory_run_opens_test_via_protocol(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="confirmatory")
    plan_dir = world.out_root / "confirmatory" / "plan" / "1"
    cfg = load_config(world.cfg_path, module="photometric")
    h = config_hash(cfg)
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), _AssertTestOpened(plan_dir / "test_opened.json"))
    assert _main("photometric")(world.argv("run")) == 5   # not in plan.lock.json
    write_plan_lock(plan_dir, "1", [{"config_hash": h, "run_id": make_run_id("photometric", "confirmatory", h, 11)}],
                    registry_path=world.registry_path)
    assert _main("photometric")(world.argv("run")) == 0
    assert load_registry(world.registry_path)[0]["test_opened"] is True
    assert "test" in set(read_predictions(_run_dir(world, "photometric", "confirmatory")).pair_id.str.split(":").str[2])


def test_pilot_runs_without_test_splits(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, profile="pilot")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    assert _main("photometric")(world.argv("run")) == 0
    splits = set(read_predictions(_run_dir(world, "photometric", "pilot")).pair_id.str.split(":").str[2])
    assert splits == {"train", "validation"}
    assert not (world.out_root / "confirmatory").exists()


def test_completed_run_of_another_command_refused(tmp_path, monkeypatch, empty_metric_params):   # review P6-I3
    from tests.fakes import BatchQueryPredictor
    world = make_world(tmp_path, monkeypatch)
    monkeypatch.setitem(registry._ALGORITHMS, ("metric", "fake"), BatchQueryPredictor())
    assert _main("metric")(world.argv("train")) == 0
    assert _main("metric")(world.argv("embed")) == 5   # same run_id (D-08 has no command): not silently skipped


def test_non_utf8_stdout_does_not_crash(monkeypatch):   # review P6-I5: cp1251 console, redirected output
    import argparse
    import io
    from expcore.cli_common import run_cli
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1251")
    monkeypatch.setattr(sys, "stdout", out)
    assert run_cli(lambda a: print("12 rows → out/метрики ✓"), [], parser=argparse.ArgumentParser()) == 0
