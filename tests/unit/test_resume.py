import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from expcore.config import RunnerCfg
from expcore.errors import ResultTamperedError, TechnicalError
from expcore.hashing import sha256_file
from expcore.io import load_json
from expcore.loaders import ImageStore, PairInput, QueryInput, SplitGuard
from expcore.resume import Runner, RunContext, predictions_content_hash, read_predictions
from expcore.runlog import RunRecord, collect_environment, make_run_id
from expcore.schemas import validate_predictions
from tests.fakes import BatchQueryPredictor, CrashOnUnitPredictor, OracleLabelPredictor, RandomScorePredictor, \
    RejectAllPredictor

pytestmark = pytest.mark.unit

H = "a" * 64
N_PAIRS = 7   # chunk_size 2 → units 0..3


@pytest.fixture
def world(tmp_path):
    rows = []
    for i in range(N_PAIRS + 1):
        rel = f"kadid10k/images/I{i:02d}.png"
        (tmp_path / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", np.full((4, 4, 3), i, np.uint8))[1].tofile(tmp_path / "data" / rel)
        rows.append(dict(image_id=f"kadid10k:{i:06d}", relative_path=rel, split="validation"))
    images = pd.DataFrame(rows)
    pairs = pd.DataFrame({"pair_id": [f"p{i}" for i in range(N_PAIRS)]})
    items = [PairInput(f"p{i}", f"kadid10k:{i:06d}", f"kadid10k:{i + 1:06d}", "P1_SAME_SCENE", {})
             for i in reversed(range(N_PAIRS))]   # unsorted on purpose
    store = ImageStore(tmp_path / "data", images, SplitGuard("smoke", fixture=True))
    return tmp_path, pairs, items, store


def make_ctx(root: Path, fixture_cfg, run_name="r"):
    cfg = dataclasses.replace(fixture_cfg, runner=RunnerCfg(chunk_size=2))
    run_id = make_run_id("photometric", "smoke", H, 11)
    run_dir = root / "out" / run_name / run_id
    record = RunRecord(run_dir, run_id=run_id, config_hash=H, dataset_manifest_hash=H, seed=11, profile="smoke",
                       module="photometric", command="run", protocol_version="1", oracle_mode=False,
                       post_test_study=False, git_commit=None, dirty_diff_hash=None,
                       environment=collect_environment(deterministic_ops=None), library_primitives=[])
    return RunContext(run_id=run_id, seed=11, profile="smoke", run_dir=run_dir, cfg=cfg, record=record)


def shards(run_dir):
    return {p.name: sha256_file(p) for p in sorted((run_dir / "predictions").glob("part-*.parquet"))}


def test_full_run_writes_valid_predictions(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    Runner(ctx, RandomScorePredictor(), store).run(items, method_id="random_score")
    pred = read_predictions(ctx.run_dir)
    validate_predictions(pred, pairs, require_complete=True)
    assert list(shards(ctx.run_dir)) == [f"part-{u:06d}.parquet" for u in range(4)]
    assert (ctx.run_dir / "_SUCCESS").is_file() and load_json(ctx.run_dir / "run.json")["status"] == "completed"
    progress = load_json(ctx.run_dir / "progress.json")
    assert sorted(progress["units"]) == ["0", "1", "2", "3"] and progress["units"]["3"]["n_rows"] == 1
    assert "comp_inliers" in pred.columns


def test_run_dir_contains_all_artifacts(world, fixture_cfg):   # §7.6 layout
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    Runner(ctx, RandomScorePredictor(), store).run(items, method_id="random_score")
    for name in ("run.json", "config.resolved.yaml", "environment.json", "rng_state.json", "file_order.json",
                 "progress.json", "run.log.jsonl", "timers.json", "augmentations.parquet", "_SUCCESS"):
        assert (ctx.run_dir / name).is_file(), name
    for name in ("run.json", "environment.json", "rng_state.json", "file_order.json", "progress.json", "timers.json"):
        load_json(ctx.run_dir / name)
    assert load_json(ctx.run_dir / "file_order.json")["keys"] == [f"p{i}" for i in range(N_PAIRS)]
    assert sorted(load_json(ctx.run_dir / "rng_state.json")["units"]) == ["0", "1", "2", "3"]
    aug = pd.read_parquet(ctx.run_dir / "augmentations.parquet")
    assert list(aug.columns) == ["unit_id", "sample_id", "aug_name", "params_json"] and len(aug) == N_PAIRS


def test_resume_skips_completed_units(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=2), store).run(items, method_id="random_score")
    run = load_json(ctx.run_dir / "run.json")
    assert run["status"] == "failed" and run["failure"]["unit_id"] == 2 and run["failure"]["type"] == "TechnicalError"
    errors = (ctx.run_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert "simulated algorithm failure" in errors
    before = shards(ctx.run_dir)
    mtimes = {p.name: p.stat().st_mtime_ns for p in (ctx.run_dir / "predictions").glob("part-*.parquet")}
    assert list(before) == ["part-000000.parquet", "part-000001.parquet"]
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    after = shards(ctx.run_dir)
    assert {k: after[k] for k in before} == before
    assert all((ctx.run_dir / "predictions" / k).stat().st_mtime_ns == t for k, t in mtimes.items())
    run = load_json(ctx.run_dir / "run.json")
    assert run["status"] == "completed" and len(run["attempts"]) == 2
    validate_predictions(read_predictions(ctx.run_dir), pairs, require_complete=True)


def test_completed_run_not_rewritten(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    Runner(ctx, RandomScorePredictor(), store).run(items, method_id="random_score")
    before, run_before = shards(ctx.run_dir), (ctx.run_dir / "run.json").read_bytes()
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    assert shards(ctx.run_dir) == before and (ctx.run_dir / "run.json").read_bytes() == run_before


def test_tampered_shard_raises(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=3), store).run(items, method_id="random_score")
    shard = ctx.run_dir / "predictions" / "part-000001.parquet"
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ResultTamperedError):
        Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    ctx2 = make_ctx(root, fixture_cfg, "done")   # completed run: tampering is detected too
    Runner(ctx2, RandomScorePredictor(), store).run(items, method_id="random_score")
    (ctx2.run_dir / "predictions" / "part-000000.parquet").write_bytes(b"junk")
    with pytest.raises(ResultTamperedError):
        Runner(make_ctx(root, fixture_cfg, "done"), RandomScorePredictor(), store).run(items, method_id="random_score")


def test_changed_items_refused_on_resume(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    with pytest.raises(ResultTamperedError, match="file_order"):
        Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items[1:], method_id="random_score")


def test_leftover_tmp_ignored(world, fixture_cfg):                      # Review Focus 3
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    junk = ctx.run_dir / "predictions" / "part-000001.parquet.tmp-123"
    junk.write_bytes(b"junk")
    assert len(read_predictions(ctx.run_dir)) == 2   # tmp file is never read as a result
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    assert not junk.exists()
    validate_predictions(read_predictions(ctx.run_dir), pairs, require_complete=True)


def test_orphan_shard_without_progress_is_recomputed(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    (ctx.run_dir / "predictions" / "part-000001.parquet").write_bytes(b"half-done unit, not in progress.json")
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    validate_predictions(read_predictions(ctx.run_dir), pairs, require_complete=True)


def test_results_independent_of_resume_history(world, fixture_cfg):
    root, pairs, items, store = world
    ctx_a = make_ctx(root, fixture_cfg, "a")
    Runner(ctx_a, RandomScorePredictor(), store).run(items, method_id="random_score")
    ctx_b = make_ctx(root, fixture_cfg, "b")
    with pytest.raises(TechnicalError):
        Runner(ctx_b, CrashOnUnitPredictor(crash_unit=0), store).run(items, method_id="random_score")
    with pytest.raises(TechnicalError):
        Runner(make_ctx(root, fixture_cfg, "b"), CrashOnUnitPredictor(crash_unit=2), store).run(
            items, method_id="random_score")
    Runner(make_ctx(root, fixture_cfg, "b"), RandomScorePredictor(), store).run(items, method_id="random_score")
    a, b = read_predictions(ctx_a.run_dir), read_predictions(ctx_b.run_dir)
    assert predictions_content_hash(a) == predictions_content_hash(b)


def test_technical_error_not_in_predictions(world, fixture_cfg):   # INV-09
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    pred = read_predictions(ctx.run_dir)
    assert set(pred.outcome) == {"decided"} and len(pred) == 2


def test_elapsed_ms_recorded(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    Runner(ctx, RejectAllPredictor(), store).run(items, method_id="reject_all")
    pred = read_predictions(ctx.run_dir)
    assert pred.elapsed_ms.notna().all() and (pred.elapsed_ms >= 0).all() and set(pred.timing_mode) == {"per_item"}
    assert set(pred.outcome) == {"reject"} and not pred.q_valid.any()
    validate_predictions(pred, pairs, require_complete=True)   # all-reject shards stay typed


def test_oracle_and_query_predictors(world, fixture_cfg):
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg, "oracle")
    Runner(ctx, OracleLabelPredictor({p.pair_id: i % 2 for i, p in enumerate(items)}), store).run(
        items, method_id="oracle_label")
    assert read_predictions(ctx.run_dir).q_valid.all()
    ctx = make_ctx(root, fixture_cfg, "query")
    queries = [QueryInput(f"kadid10k:{i:06d}", {}) for i in range(5)]
    Runner(ctx, BatchQueryPredictor(), store).run(queries, method_id="batch_query")
    pred = read_predictions(ctx.run_dir)
    assert pred.pair_id.isna().all() and set(pred.timing_mode) == {"amortized_batch"} and len(pred) == 5
    validate_predictions(pred, pd.DataFrame({"query_id": [q.query_id for q in queries]}), require_complete=True)


def test_cli_started_record_is_not_restarted(world, fixture_cfg):   # review P5-I1
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    ctx.record.start()   # e.g. the CLI starts the record before looking up the algorithm
    Runner(ctx, RandomScorePredictor(), store).run(items, method_id="random_score")
    assert [a["status"] for a in load_json(ctx.run_dir / "run.json")["attempts"]] == ["completed"]


def test_kill_between_success_and_finish_is_repaired(world, fixture_cfg, monkeypatch):   # review P5-I2
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    real_finish = RunRecord.finish

    def killed(self, status, failure=None):
        if status == "completed":
            raise KeyboardInterrupt   # process dies right after _SUCCESS
        real_finish(self, status, failure)
    monkeypatch.setattr(RunRecord, "finish", killed)
    with pytest.raises(KeyboardInterrupt):
        Runner(ctx, RandomScorePredictor(), store).run(items, method_id="random_score")
    monkeypatch.setattr(RunRecord, "finish", real_finish)
    assert (ctx.run_dir / "_SUCCESS").exists() and load_json(ctx.run_dir / "run.json")["status"] == "running"
    before = shards(ctx.run_dir)
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    assert load_json(ctx.run_dir / "run.json")["status"] == "completed" and shards(ctx.run_dir) == before


def test_augmentations_not_duplicated_on_recompute(tmp_path):   # review P5-I3
    from expcore.resume import AugmentationLogger
    log = AugmentationLogger(tmp_path)
    log.log(0, "p0", "flip", {}); log.commit(0)
    log.log(0, "p0", "flip", {}); log.commit(0)   # unit 0 recomputed after a crash before progress.json
    log.log(1, "p1", "flip", {}); log.commit(1)
    aug = pd.read_parquet(tmp_path / "augmentations.parquet")
    assert sorted(zip(aug.unit_id, aug.sample_id)) == [(0, "p0"), (1, "p1")]


def test_attempts_keep_their_provenance(world, fixture_cfg):   # review P5-I4
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    ctx2 = make_ctx(root, fixture_cfg)
    ctx2.record._static["git_commit"] = "c2"
    Runner(ctx2, RandomScorePredictor(), store).run(items, method_id="random_score")
    attempts = load_json(ctx.run_dir / "run.json")["attempts"]
    assert [a["git_commit"] for a in attempts] == [None, "c2"]
    for k, a in enumerate(attempts):
        env = ctx.run_dir / a["environment_path"]
        assert env.name == f"environment.attempt-{k}.json" and sha256_file(env) == a["environment_sha256"]


def test_timers_record_every_attempt_and_peak_memory(world, fixture_cfg):   # review P5-I8
    root, pairs, items, store = world
    ctx = make_ctx(root, fixture_cfg)
    with pytest.raises(TechnicalError):
        Runner(ctx, CrashOnUnitPredictor(crash_unit=1), store).run(items, method_id="random_score")
    Runner(make_ctx(root, fixture_cfg), RandomScorePredictor(), store).run(items, method_id="random_score")
    attempts = load_json(ctx.run_dir / "timers.json")["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "completed"]
    assert all("peak_memory_mb" in a and "peak_memory_reason" in a for a in attempts)
