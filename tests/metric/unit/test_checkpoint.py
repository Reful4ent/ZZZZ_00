import json

import numpy as np
import pandas as pd
import pytest

from tests.metric.smoke.fixtures import make_sop_like, smoke_params

pytestmark = [pytest.mark.tf, pytest.mark.filterwarnings("ignore::DeprecationWarning")]

N_CLS, K, D, CROP = 8, 4, 16, 56
STATE = {"epoch": 1, "global_step": 2, "seed": 7, "sampler_version": "p3-sampler-v1",
         "best_epoch": None, "best_correct_at_1": None, "config_hash": "c", "manifest_hash": "m",
         "weights_sha256": None}


def _objects(seed):
    from metric.model import build_encoder
    from metric.proxies import init_proxies, make_optimizer
    from metric.queues import ClassQueues
    from metric.train_step import StepConfig, make_proxy_step
    enc = build_encoder(None, CROP, D, seed)
    P = init_proxies(N_CLS, D, np.random.default_rng(seed))
    q, opt = ClassQueues(N_CLS, K, D), make_optimizer(1e-3)
    step = make_proxy_step(enc, P, q, opt, StepConfig(10.0, 0.5, 1, 1e-12, 1e-12))
    return enc, P, q, opt, step


def _vars(enc, P, q, opt):
    return [*enc.variables, P, q.buf, q.start, q.n, q.S, q.ins_step, *opt.variables]


@pytest.fixture(scope="module")
def objs():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    a = _objects(1)
    rng = np.random.default_rng(0)
    for t in range(3):
        x = rng.uniform(0, 255, (8, CROP, CROP, 3)).astype("float32")
        a[4](x, np.array([0, 1, 2, 3, 0, 1, 2, 3], np.int32), tf.constant(t, tf.int64))
    return a[:4], _objects(2)[:4]


def test_roundtrip_all_state(objs, tmp_path):
    from metric.checkpoint import restore_full, save_full
    a, b = objs
    assert int(a[3].iterations) == 3 and int(a[2].n.numpy().max()) > 0
    save_full(tmp_path / "last", *a, STATE)
    assert restore_full(tmp_path / "last", *b) == STATE
    assert int(b[3].iterations) == 3
    for va, vb in zip(_vars(*a), _vars(*b), strict=True):
        np.testing.assert_array_equal(va.numpy(), vb.numpy())
    rng = json.loads((tmp_path / "last" / "rng_state.json").read_text(encoding="utf-8"))
    assert "numpy_global" in rng and "python_random" in rng


def test_interrupted_save_keeps_last(objs, tmp_path, monkeypatch):
    import metric.checkpoint as ck
    a, b = objs
    d = tmp_path / "last"
    ck.save_full(d, *a, STATE)
    with monkeypatch.context() as mp:
        mp.setattr(ck, "dump_json", lambda *_: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError):
            ck.save_full(d, *a, {**STATE, "global_step": 5})
    assert (tmp_path / "last.tmp").exists()
    assert ck.restore_full(d, *b)["global_step"] == 2
    ck.save_full(d, *a, {**STATE, "global_step": 7})
    assert not (tmp_path / "last.tmp").exists() and not (tmp_path / "last.old").exists()
    d.rename(tmp_path / "last.old")  # crash between the two renames of the swap
    assert ck.restore_full(d, *b)["global_step"] == 7
    real_rmtree = ck.shutil.rmtree
    with monkeypatch.context() as mp:  # best-effort cleanup fails silently (Windows file handle)
        mp.setattr(ck.shutil, "rmtree",
                   lambda p, ignore_errors=False: None if ignore_errors else real_rmtree(p))
        ck.save_full(d, *a, {**STATE, "global_step": 8})
        assert (tmp_path / "last.old").exists()  # stale .old left behind
        ck.save_full(d, *a, {**STATE, "global_step": 9})
    real_rmtree(tmp_path / "last.old")
    assert ck.restore_full(d, *b)["global_step"] == 9


def test_check_drift_raises_with_details(objs):
    from metric.errors import RunStop
    from metric.train import _check_drift
    q = objs[0][2]
    ok = _check_drift(q, 1e-6, 10)
    assert ok["global_step"] == 10 and ok["max_abs_diff"] <= 1e-6 and ok["sum_tol"] == 1e-6
    S0 = q.S.numpy()
    try:
        q.S.assign(S0 + np.where(np.arange(N_CLS)[:, None] == 1, 0.5, 0.0))
        with pytest.raises(RunStop) as e:
            _check_drift(q, 1e-6, 11)
    finally:
        q.S.assign(S0)
    assert e.value.failure_type == "QUEUE_SUM_DRIFT"
    assert e.value.details["global_step"] == 11 and e.value.details["sum_tol"] == 1e-6
    assert e.value.details["max_abs_diff"] == pytest.approx(0.5)


def _failing_run(tmp_path, params, fail_at, inject):
    """run_training with the step wrapped: call number `fail_at` goes through `inject`."""
    import metric.train as train
    images, store = make_sop_like()
    real, calls = train.make_proxy_step, []

    def spy(encoder, P, queues, optimizer, cfg):
        fn = real(encoder, P, queues, optimizer, cfg)
        calls.append({"objs": (encoder, P, queues, optimizer)})

        def step(x, y, t):
            calls.append(int(x.shape[0]))
            return inject(fn, x, y, t) if len(calls) - 2 == fail_at else fn(x, y, t)
        return step

    ctx = train.TrainContext(params=params, seed=7, run_dir=tmp_path / "run", images=images,
                             store=store)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "make_proxy_step", spy)
        res = train.run_training(ctx)
    return res, ctx.run_dir, calls


def test_invalid_norm_stop_recorded(tmp_path):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from metric.checkpoint import load_progress, restore_full
    from metric.steplog import read_steps
    p = smoke_params(lam=0.1, epochs=2, steps_per_epoch=3)
    p["checkpoint"]["every_steps"] = 2
    nan = lambda fn, x, y, t: fn(tf.fill(tf.shape(x), float("nan")), y, t)  # noqa: E731
    res, d, calls = _failing_run(tmp_path, p, 4, nan)
    assert res["status"] == "failed" and res["failure"]["type"] == "INVALID_EMBEDDING_NORM"
    f = json.loads((d / "failure.json").read_text(encoding="utf-8"))
    assert f["failure_type"] == "INVALID_EMBEDDING_NORM" and f["details"]["global_step"] == 4
    ids = f["details"]["image_ids"]
    assert len(ids) == 32 and set(ids) <= set(make_sop_like()[0].query("split == 'train'")["image_id"])
    assert len(calls) == 1 + 5  # no retry
    st = json.loads((d / "checkpoints/last/state.json").read_text(encoding="utf-8"))
    assert st["global_step"] == 3 and st["epoch"] == 2 and st["best_epoch"] == 1
    assert restore_full(d / "checkpoints/last", *calls[0]["objs"])["global_step"] == 3
    part = pd.read_parquet(d / "training_steps/part-002.parquet")
    assert list(part["global_step"]) == [3]  # partial epoch rows are kept
    rows, prog = load_progress(d / "checkpoints/last")  # epoch 2 progress saved with step 3
    assert rows == read_steps(d / "training_steps/part-002.parquet")
    assert prog["losses"] == [[part["L_base"][0], part["L_dev"][0], part["L_total"][0]]]
    assert [c["global_step"] for c in res["queue_checks"]] == [1, 2, 3]


def test_oom_recorded(tmp_path):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    def oom(fn, x, y, t):
        raise tf.errors.ResourceExhaustedError(None, None, "OOM when allocating tensor")

    res, d, calls = _failing_run(tmp_path, smoke_params(epochs=1, steps_per_epoch=3), 1, oom)
    assert res["status"] == "failed" and res["failure"]["type"] == "OOM"
    assert res["failure"]["details"]["global_step"] == 1
    assert calls[1:] == [32, 32]  # batch size unchanged, no retry
    f = json.loads((d / "failure.json").read_text(encoding="utf-8"))
    assert f["failure_type"] == "OOM" and "peak_gpu_mem_bytes" in f["details"]
    assert not (d / "checkpoints/last").exists()
    assert list(pd.read_parquet(d / "training_steps/part-001.parquet")["global_step"]) == [0]
