"""Run the whole smoke pipeline in this (fresh) process and print a JSON result line (T-REP-02).

Usage: python -m tests.smoke.smoke_driver <root>
Registers the test doubles and toy pair builders, then build-manifests → validate-data → plan → run →
aggregate → report, all through experiments.cli.main.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

from expcore import registry
from expcore.resume import predictions_content_hash, read_predictions
from experiments.cli import main
from experiments.plan import load_plan
from tests.cli_helpers import make_world
from tests.conftest import write_cfg
from tests.fakes import RandomScorePredictor, RejectAllPredictor

STEPS = ("build-manifests", "validate-data", "plan", "run", "aggregate", "report")


def run_pipeline(root: Path, monkeypatch) -> dict:
    world = make_world(root, monkeypatch, build=False, runs=[
        {"module": "photometric", "command": "run", "config": "cfg.yaml"},
        {"module": "photometric", "command": "run", "config": "cfg_reject.yaml"}])
    write_cfg({**copy.deepcopy(world.cfg_dict), "algorithms": {"reject_all": "1"}}, root / "cfg_reject.yaml")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "reject_all"), RejectAllPredictor())
    codes = {step: main(world.exp_argv(step)) for step in STEPS}
    plan_dir = world.out_root / "smoke" / "plan" / "1"
    hashes = {r.run_id: predictions_content_hash(read_predictions(world.out_root / "smoke" / r.module / r.run_id))
              for r in load_plan(plan_dir)} if codes["plan"] == 0 else {}
    return {"codes": codes, "hashes": hashes,
            "report": str(world.out_root / "smoke" / "report" / "1" / "report.md")}


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as mp:
        result = run_pipeline(Path(sys.argv[1]), mp)
    print("RESULT " + json.dumps(result, ensure_ascii=True))
