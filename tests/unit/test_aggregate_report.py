import copy

import pandas as pd
import pytest

from expcore import registry
from expcore.errors import DataValidationError, ProtocolViolationError
from expcore.hashing import sha256_file
from expcore.io import dump_json, load_json, write_parquet
from expcore.reporting import DISCLAIMER, dataset_display_name
from expcore.schemas import PREDICTIONS_COLUMNS, PREDICTIONS_NULLABLE
from experiments.aggregate import aggregate
from experiments.cli import main
from experiments.plan import load_plan
from experiments.report import render_report
from tests.cli_helpers import make_world
from tests.conftest import write_cfg
from tests.fakes import RandomScorePredictor, RejectAllPredictor

pytestmark = pytest.mark.unit   # T-AGG-01, T-REP-01


def _build(tmp_path, monkeypatch):
    """pilot SOP world, three plan entries: a scorer, a reject-everything baseline, an oracle run."""
    w = make_world(tmp_path, monkeypatch, profile="pilot", runs=[
        {"module": "photometric", "command": "run", "config": "cfg.yaml"},
        {"module": "photometric", "command": "run", "config": "cfg_reject.yaml"},
        {"module": "photometric", "command": "run", "config": "cfg_oracle.yaml"}])
    write_cfg({**copy.deepcopy(w.cfg_dict), "algorithms": {"reject_all": "1"}}, tmp_path / "cfg_reject.yaml")
    oracle = copy.deepcopy(w.cfg_dict)
    oracle["data_modes"]["oracle"] = {"enabled": True, "reason": "upper bound"}
    write_cfg(oracle, tmp_path / "cfg_oracle.yaml")
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "fake"), RandomScorePredictor())
    monkeypatch.setitem(registry._ALGORITHMS, ("photometric", "reject_all"), RejectAllPredictor())
    assert main(w.exp_argv("plan")) == 0 and main(w.exp_argv("run")) == 0
    w.plan_dir = w.out_root / "pilot" / "plan" / "1"
    return w


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """Read-only world shared by the tests that do not modify run directories."""
    with pytest.MonkeyPatch.context() as mp:
        yield _build(tmp_path_factory.mktemp("aggregate"), mp)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A private world for tests that tamper with run directories."""
    return _build(tmp_path, monkeypatch)


def _metrics(world):
    return aggregate(world.plan_dir, world.root / "metrics.parquet")


def test_aggregate_counts_independent_units(shared):
    world = shared
    m = _metrics(world)
    main_rows = m[(m.section == "main") & (m.method_id == "fake")]
    assert (main_rows.n_source_groups < main_rows.n_pairs).all() and set(main_rows.n_seeds) == {3}
    assert set(main_rows.seed) == {11, 23, 37} and set(main_rows.split) == {"train", "validation"}
    assert set(m.relation_id) == {"P1_SAME_SCENE"} and set(m.score_transform) == {"identity"}
    assert (world.root / "metrics.parquet").is_file()


def test_oracle_runs_separated(shared):
    world = shared
    m = _metrics(world)
    assert set(m[m.section == "oracle"].method_id) == {"fake+oracle"}
    assert "fake+oracle" not in set(m[m.section == "main"].method_id)
    text = render_report(m, load_plan(world.plan_dir), _completeness(world))
    main_part, oracle_part = text.split("## Oracle-запуски")
    assert "fake+oracle" not in main_part and "fake+oracle" in oracle_part


def test_aggregate_rejects_incomplete_predictions(world):   # Review Focus 2
    run = load_plan(world.plan_dir)[0]
    run_dir = world.out_root / "pilot" / "photometric" / run.run_id
    shard = run_dir / "predictions" / "part-000000.parquet"
    df = pd.read_parquet(shard).iloc[1:]   # one pair silently missing, shard kept consistent with progress.json
    comps = [c for c in df.columns if c.startswith("comp_")]
    write_parquet(df, shard, schema={**PREDICTIONS_COLUMNS, **dict.fromkeys(comps, "float64")},
                  nullable=PREDICTIONS_NULLABLE | frozenset(comps))
    progress = load_json(run_dir / "progress.json")
    progress["units"]["0"].update(n_rows=len(df), sha256=sha256_file(shard))
    dump_json(progress, run_dir / "progress.json")
    with pytest.raises(DataValidationError, match="INV-07"):
        _metrics(world)
    assert main(world.exp_argv("aggregate")) == 3


def test_aggregate_detects_tampered_shard(world):
    run = load_plan(world.plan_dir)[0]
    shard = world.out_root / "pilot" / "photometric" / run.run_id / "predictions" / "part-000000.parquet"
    shard.write_bytes(shard.read_bytes() + b"x")
    assert main(world.exp_argv("aggregate")) == 5


def _completeness(world):
    plan = load_plan(world.plan_dir)
    runs = {r.run_id: load_json(world.out_root / "pilot" / r.module / r.run_id / "run.json")["status"] for r in plan}
    return {"runs": runs, "datasets": {"sop": None}}


def test_report_has_disclaimer_matrix_and_class_coverage(shared):
    world = shared
    text = render_report(_metrics(world), load_plan(world.plan_dir), _completeness(world))
    assert DISCLAIMER in text
    for needle in ("reject", "Coverage+", "Coverage−", "Recall_all", "Recall_cond (conditional)",
                   "Изменчивость по seeds", "Stanford Online Products"):
        assert needle in text, needle


def test_report_includes_worse_than_baseline_runs(shared):
    world = shared
    text = render_report(_metrics(world), load_plan(world.plan_dir), _completeness(world))
    assert "reject_all" in text
    for r in load_plan(world.plan_dir):
        assert r.run_id in text


def test_report_fails_if_run_missing(shared):
    world = shared
    m = _metrics(world)
    dropped = m[m.run_id != load_plan(world.plan_dir)[0].run_id]
    with pytest.raises(ProtocolViolationError):
        render_report(dropped, load_plan(world.plan_dir), _completeness(world))


def test_report_cli_writes_markdown(shared):
    world = shared
    assert main(world.exp_argv("aggregate")) == 0 and main(world.exp_argv("report")) == 0
    report = world.out_root / "pilot" / "report" / "1" / "report.md"
    assert DISCLAIMER in report.read_text(encoding="utf-8")


def test_hpatches_subset_named_in_report(shared):
    world = shared   # T-DAT-07
    comp = _completeness(world)
    comp["datasets"]["hpatches"] = {"status": "subset", "subset_name": "mini"}
    text = render_report(_metrics(world), load_plan(world.plan_dir), comp)
    assert "HPatches, поднабор `mini`" in text
    assert dataset_display_name("hpatches", {"status": "full"}) == "HPatches"


def test_report_states_positive_class(shared):
    world = shared
    text = render_report(_metrics(world), load_plan(world.plan_dir), _completeness(world))
    assert "Положительный класс: match (y_match = 1)" in text and "identity" in text
