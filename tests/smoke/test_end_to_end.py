import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from expcore import registry
from expcore.io import load_json, read_parquet
from expcore.reporting import DISCLAIMER
from experiments.cli import main
from experiments.plan import load_plan
from tests.cli_helpers import make_world
from tests.fakes import RandomScorePredictor
from tests.smoke.smoke_driver import STEPS

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]


def run_smoke_in_subprocess(root: Path) -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run([sys.executable, "-m", "tests.smoke.smoke_driver", str(root)], cwd=REPO, env=env,
                         capture_output=True, text=True, encoding="utf-8")
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, f"driver failed (exit {res.returncode}):\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
    return json.loads(line[len("RESULT "):])


def test_pipeline_build_validate_plan_run_aggregate_report(tmp_path):   # §11.3 п.2, 6
    result = run_smoke_in_subprocess(tmp_path)
    assert result["codes"] == dict.fromkeys(STEPS, 0)
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert DISCLAIMER in report and "| Истинный класс | y_pred=1 | y_pred=0 | reject |" in report
    assert "Coverage+" in report and "reject_all" in report   # results worse than any baseline are not filtered


def test_two_fresh_process_runs_identical_predictions(tmp_path):   # T-REP-02, §11.3 п.3
    a = run_smoke_in_subprocess(tmp_path / "a")
    b = run_smoke_in_subprocess(tmp_path / "b")
    assert a["hashes"] and a["hashes"] == b["hashes"]


def test_corrupted_fixture_makes_validate_exit_3(tmp_path, monkeypatch):   # §11.3 п.2
    world = make_world(tmp_path, monkeypatch, build=False)
    assert main(world.exp_argv("build-manifests")) == 0
    images = read_parquet(Path(world.cfg_dict["paths"]["manifest_dir"]) / "v1" / "images.parquet")
    victim = Path(world.cfg_dict["paths"]["data_root"]) / images.relative_path.iloc[5]
    victim.write_bytes(victim.read_bytes()[:40])   # truncated image
    assert main(world.exp_argv("validate-data")) == 3
    fresh = make_world(tmp_path / "fresh", monkeypatch, build=False)
    first = sorted((Path(fresh.cfg_dict["paths"]["data_root"]) / "kadid10k" / "images").iterdir())[0]
    first.write_bytes(b"not an image")
    assert main(fresh.exp_argv("build-manifests")) == 3   # corrupted delivery: no manifest


def test_oracle_marked_everywhere(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, build=False)
    world.cfg_dict["data_modes"]["oracle"] = {"enabled": True, "reason": "upper bound"}
    world.write()
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    for step in STEPS:
        assert main(world.exp_argv(step)) == 0, step
    (run,) = load_plan(world.out_root / "smoke" / "plan" / "1")
    run_dir = world.out_root / "smoke" / "photometric" / run.run_id
    assert load_json(run_dir / "run.json")["oracle_mode"] is True
    metrics = read_parquet(world.out_root / "smoke" / "aggregate" / "1" / "metrics.parquet")
    assert set(metrics.method_id) == {"fake+oracle"} and set(metrics.section) == {"oracle"}
    report = (world.out_root / "smoke" / "report" / "1" / "report.md").read_text(encoding="utf-8")
    main_part, oracle_part = report.split("## Oracle-запуски")
    assert "fake+oracle" in oracle_part and "fake+oracle" not in main_part


class _PeekAtTest(RandomScorePredictor):
    """A misbehaving algorithm that tries to read a test image during a pilot run."""

    def __init__(self, test_image_id):
        self._test_image_id = test_image_id

    def predict(self, pair, images, ctx):
        images.load(self._test_image_id, "gray")
        return super().predict(pair, images, ctx)


def test_pilot_profile_cannot_touch_test_split_e2e(tmp_path, monkeypatch):   # T-REP-04
    world = make_world(tmp_path, monkeypatch, profile="pilot", build=False)
    assert main(world.exp_argv("build-manifests")) == 0
    images = read_parquet(Path(world.cfg_dict["paths"]["manifest_dir"]) / "v1" / "images.parquet")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"),
                        _PeekAtTest(images[images.split == "test"].image_id.iloc[0]))
    assert main(world.exp_argv("plan")) == 0
    assert main(world.exp_argv("run")) == 5   # TestAccessError
    (run, *_) = load_plan(world.out_root / "pilot" / "plan" / "1")
    failure = load_json(world.out_root / "pilot" / "photometric" / run.run_id / "run.json")["failure"]
    assert failure["type"] == "TestAccessError"
