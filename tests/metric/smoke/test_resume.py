"""ПРИЁМ-М3 (spec 7.5-7.6): 12 steps uninterrupted == 5 steps, new process, resume, 7 more."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.tf,
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]

ROOT = Path(__file__).resolve().parents[3]
TIMING = ["step_time_ms", "data_time_ms", "epoch_time_s", "val_time_s"]


def _proc(run_dir, mode, variant):
    subprocess.run([sys.executable, "-m", "tests.metric.smoke._resume_entry", str(run_dir), mode,
                    variant], cwd=ROOT, check=True, timeout=3600)


def _fingerprint(path):
    return path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def make_runs(tmp_path_factory, variant):
    """Run A uninterrupted; run B cut after 5 steps, then resumed in a new process."""
    a = tmp_path_factory.mktemp("a") / "run"  # same dir name -> same run_id
    b = tmp_path_factory.mktemp("b") / "run"
    _proc(a, "train", variant)
    _proc(b, "train-5", variant)
    part1 = _fingerprint(b / "training_steps/part-001.parquet")
    _proc(b, "resume", variant)
    return {"a": a, "b": b, "part1": part1}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    pytest.importorskip("tensorflow")
    return make_runs(tmp_path_factory, "smoke")


def _ckpt(run_dir):
    import tensorflow as tf
    r = tf.train.load_checkpoint(str(run_dir / "checkpoints/last/ckpt"))
    return {k: r.get_tensor(k) for k in r.get_variable_to_shape_map() if "OBJECT_GRAPH" not in k}


def _frame(path):
    df = pd.read_parquet(path)
    return df.drop(columns=[c for c in TIMING if c in df.columns])


def test_10_equals_5_restore_5(runs):
    assert_same_runs(runs)


def assert_same_runs(runs):
    a, b = _ckpt(runs["a"]), _ckpt(runs["b"])
    assert a.keys() == b.keys()
    assert any(k.startswith("queues/") for k in a) and any(k.startswith("P/") for k in a)
    for k in a:
        if k.startswith("queues/") or "iter" in k:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            np.testing.assert_allclose(a[k], b[k], atol=1e-6, err_msg=k)
    it = [k for k in a if k.startswith("optimizer/") and "iter" in k]
    assert it and all(int(a[k]) == int(b[k]) == 12 for k in it)
    for f in ("batches.parquet", "epochs.parquet", "training_steps.parquet"):
        pd.testing.assert_frame_equal(_frame(runs["a"] / f), _frame(runs["b"] / f), obj=f)
    steps = pd.read_parquet(runs["b"] / "training_steps.parquet")
    assert list(steps["global_step"]) == list(range(12))
    assert (runs["a"] / "best_checkpoint.json").read_text() == \
        (runs["b"] / "best_checkpoint.json").read_text()


def test_completed_parts_not_rewritten(runs):
    assert _fingerprint(runs["b"] / "training_steps/part-001.parquet") == runs["part1"]


@pytest.mark.parametrize("field", ["config_hash", "manifest_hash", "weights_sha256"])
def test_resume_rejects_mismatch(runs, field):
    from expcore.errors import ConfigError
    from metric.train import resume_training
    from tests.metric.smoke._resume_entry import context
    ctx = context(runs["b"], **{field: "edited"})  # e.g. params.calibration.lambda changed
    with pytest.raises(ConfigError, match=field):
        resume_training(ctx, runs["b"] / "checkpoints/last")
