"""Task 7.1 e2e: build-manifests (p3 builder) -> train -> embed val/test -> search --pairs val/test -> select."""
import copy

import numpy as np
import pandas as pd
import pytest

from expcore.io import load_json
from expcore.manifests import load_manifest
from expcore.resume import read_predictions, verify_shards
from expcore.schemas import validate_predictions, validate_run_json
from tests.metric.unit.test_cli import SPEC_DISCLAIMERS, _main, _select, _train_run_dir
from tests.metric.world import SMOKE_PARAMS, metric_world

pytestmark = [pytest.mark.smoke, pytest.mark.tf, pytest.mark.filterwarnings("ignore::DeprecationWarning")]
COMMON = ("run.json", "config.resolved.yaml", "environment.json", "rng_state.json", "run.log.jsonl", "timers.json",
          "_SUCCESS")


def _stage_dir(world, **match):
    found = [p.parent for p in (world.out_root / "smoke" / "metric").glob("*/stage.json")
             if all(load_json(p)["stage"].get(k) == v for k, v in match.items())]
    assert len(found) == 1, found
    return found[0]


def _check_run(run_dir, command):
    for name in COMMON:
        assert (run_dir / name).exists(), name
    run = load_json(run_dir / "run.json")
    validate_run_json(run)
    assert run["status"] == "completed" and run["command"] == command


def test_e2e_smoke(tmp_path, monkeypatch, capsys):
    pytest.importorskip("tensorflow")
    params = copy.deepcopy(SMOKE_PARAMS)
    params["train"].update(epochs=2, steps_per_epoch=2)
    from metric.params import BASE_PARAMS
    monkeypatch.setitem(BASE_PARAMS["train"], "epochs", 2)   # runtime: the profile fixes 40 epochs even in smoke
    world = metric_world(tmp_path, monkeypatch, params=params, sop_classes=25, real_pairs=True)
    manifest = load_manifest(world.out_root / "manifests", "v1")
    pairs = manifest.pairs["p3"]

    assert _main(world.argv("train")) == 0
    train_dir = _train_run_dir(world)
    _check_run(train_dir, "train")
    for name in ("checkpoints/last/state.json", "checkpoints/best/encoder.keras", "training_steps.parquet",
                 "epochs.parquet", "queue_state/epoch-001.parquet", "batches.parquet", "proxy_index.parquet",
                 "sampler_exclusions.parquet", "best_checkpoint.json", "memory.json"):
        assert (train_dir / name).exists(), name
    best = load_json(train_dir / "best_checkpoint.json")
    assert all(t in best["disclaimers"] for t in SPEC_DISCLAIMERS)
    assert load_json(train_dir / "memory.json")["total_static"] > 0
    steps = pd.read_parquet(train_dir / "training_steps.parquet")
    assert len(steps) == 4

    assert _main(world.argv("train")) == 0   # completed: not recomputed
    assert len(load_json(train_dir / "run.json")["attempts"]) == 1
    assert _main(["resume", "--checkpoint", str(train_dir / "checkpoints" / "last"),
                  "--protocol-registry", str(world.registry_path)]) == 5   # completed runs are never resumed

    for split in ("validation", "test"):
        assert _main(world.argv("embed", "--split", split)) == 0
    emb = {s: _stage_dir(world, command="embed", split=s) for s in ("validation", "test")}
    for split, d in emb.items():
        _check_run(d, "embed")
        with np.load(d / "embeddings.npz") as z:
            assert {"embeddings", "image_ids", "weights_hash", "split", "degradation_id", "method_id", "d"} <= set(z)
            assert str(z["split"]) == split and z["embeddings"].dtype == np.float32
        assert load_json(d / "stage.json")["base_run_id"] == train_dir.name

    def search(split, *extra):
        npz = str(emb[split] / "embeddings.npz")
        return _main(world.argv("search", "--gallery", npz, "--queries", npz, "--pairs", *extra))

    assert search("validation") == 0
    val = _stage_dir(world, command="search", split="validation")
    _check_run(val, "search")
    assert (val / "thresholds.json").exists()
    assert search("test") == 5   # test needs the validation threshold
    assert search("test", "--thresholds", str(val / "thresholds.json")) == 0
    test = _stage_dir(world, command="search", split="test")
    _check_run(test, "search")
    for d, split in ((val, "validation"), (test, "test")):
        for name in ("retrieval.parquet", "retrieval_summary.json", "pair_scores.parquet"):
            assert (d / name).exists(), name
        verify_shards(d)
        pred = read_predictions(d)
        split_pairs = pairs[pairs["split"] == split]
        validate_predictions(pred, split_pairs, require_complete=True)
        assert set(pred["pair_id"]) == set(split_pairs["pair_id"])

    code, out, _ = _select(world, [4, 6])
    assert code == 0 and all(t in load_json(out)["disclaimers"] for t in SPEC_DISCLAIMERS)
