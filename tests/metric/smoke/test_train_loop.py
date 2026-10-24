import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tests.metric.smoke.fixtures import make_sop_like, smoke_params

pytestmark = [pytest.mark.smoke, pytest.mark.tf,
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _run(tmp_path_factory, name, params, fake_val=None):
    """run_training with make_proxy_step wrapped to capture θ₀/P₀ and the live objects."""
    pytest.importorskip("tensorflow")
    import metric.train as train
    from metric.model import weights_hash
    images, store = make_sop_like()
    cap = {}
    real = train.make_proxy_step

    def spy(encoder, P, queues, optimizer, cfg):
        cap.update(theta0=weights_hash(encoder, ""), P0=P.numpy().copy(), encoder=encoder, P=P,
                   queues=queues, opt=optimizer)
        return real(encoder, P, queues, optimizer, cfg)

    ctx = train.TrainContext(params=params, seed=7, run_dir=tmp_path_factory.mktemp(name),
                             images=images, store=store, method="proxy_cal")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "make_proxy_step", spy)
        if fake_val is not None:
            seq = iter(fake_val)
            mp.setattr(train, "validate_epoch", lambda *a: {
                "val_recall_at_1": 0.5, "val_recall_at_5": 0.5, "val_recall_at_10": 0.5,
                "val_correct_at_1": next(seq), "n_queries": 10, "n_valid_queries": 10,
                "n_no_positive": 0})
        cap["result"] = train.run_training(ctx)
    cap.update(ctx=ctx, images=images, store=store)
    return cap


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return _run(tmp_path_factory, "smoke", smoke_params(lam=0.1))


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Three 3-epoch × 1-step runs; validation is faked as [5, 7, 7] correct."""
    p = lambda **kw: smoke_params(epochs=3, steps_per_epoch=1, **kw)  # noqa: E731
    return {k: _run(tmp_path_factory, k, p(**kw), fake_val=[5, 7, 7]) for k, kw in {
        "lam0": {"lam": 0.0}, "lam01": {"lam": 0.1}, "minimal": {"lam": 0.0, "queues": False},
    }.items()}


def test_train_loop_smoke(smoke):
    d = smoke["ctx"].run_dir
    assert smoke["result"]["status"] == "completed" and smoke["result"]["total_steps"] == 6
    assert smoke["result"]["deterministic_ops"] is True
    steps = pd.read_parquet(d / "training_steps.parquet")
    assert list(steps["global_step"]) == list(range(6)) and list(steps["epoch"]) == [1] * 3 + [2] * 3
    assert steps["L_dev"].iloc[0] == 0.0 and steps["n_cval"].iloc[0] == 0
    assert steps["lambda"].eq(0.1).all() and steps["n_c_max"].iloc[0] == 2
    ep = pd.read_parquet(d / "epochs.parquet")
    assert list(ep["epoch"]) == [1, 2] and list(ep["global_step"]) == [2, 5]
    assert (ep["n_queries"] == (smoke["images"]["split"] == "validation").sum()).all()
    assert len(pd.read_parquet(d / "batches.parquet")) == 6 * 32
    best = json.loads((d / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert best["epoch"] in (1, 2)
    assert best["selection_rule"] == "max_val_recall_at_1_tie_earliest"
    assert best["encoder_path"] == "checkpoints/best/encoder.keras"
    import tensorflow as tf
    assert tf.keras.models.load_model(d / best["encoder_path"]).output_shape == (None, 128)
    pre = json.loads((d / "checkpoints/best/preprocess.json").read_text(encoding="utf-8"))
    assert pre == {"resize_short": 64, "crop": 56, "channel_order": "RGB",
                   "preprocess": "tf.keras.applications.resnet50.preprocess_input",
                   "embedding_dim": 128, "normalize_l2": True}
    st = json.loads((d / "checkpoints/last/state.json").read_text(encoding="utf-8"))
    assert (st["epoch"], st["global_step"], st["sampler_version"]) == (2, 5, "p3-sampler-v1")
    assert st["best_epoch"] == best["epoch"]
    from metric.checkpoint import load_progress
    assert load_progress(d / "checkpoints/last") == ([], {"losses": []})  # epoch end: flushed
    assert [c["global_step"] for c in smoke["result"]["queue_checks"]] == [2, 5]
    for f in ("proxy_index.parquet", "sampler_exclusions.parquet", "queue_state/epoch-001.parquet",
              "queue_state/epoch-002.parquet", "training_steps/part-002.parquet"):
        assert (d / f).exists(), f
    qs = pd.read_parquet(d / "queue_state/epoch-002.parquet")
    assert list(qs.columns) == ["proxy_idx", "n_c", "mean_age", "mu_norm", "p_norm"]


def _state_hash(enc, P, q, opt):
    h = hashlib.sha256()
    for v in [*enc.variables, P, q.buf, q.start, q.n, q.S, q.ins_step, *opt.variables]:
        h.update(np.ascontiguousarray(v.numpy()).tobytes())
    return h.hexdigest()


def test_validation_does_not_mutate(smoke):
    from metric.train import validate_epoch
    enc, P, q, opt = (smoke[k] for k in ("encoder", "P", "queues", "opt"))
    assert int(opt.iterations) == 6
    before = _state_hash(enc, P, q, opt)
    imgs = smoke["images"]
    out = validate_epoch(enc, "proxy_cal", imgs[imgs["split"] == "validation"], smoke["store"],
                         smoke["ctx"].params)
    assert 0 <= out["val_correct_at_1"] <= out["n_valid_queries"]
    assert _state_hash(enc, P, q, opt) == before


def test_best_epoch_tie_earlier(pair):
    r = pair["lam0"]
    assert r["result"]["best_epoch"] == 2 and r["result"]["best_correct_at_1"] == 7
    best = json.loads((r["ctx"].run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert best["epoch"] == 2 and best["global_step"] == 1 and best["val_correct_at_1"] == 7


def test_paired_runs_same_start(pair):
    a, b = pair["lam0"], pair["lam01"]
    assert a["theta0"] == b["theta0"]
    np.testing.assert_array_equal(a["P0"], b["P0"])
    pd.testing.assert_frame_equal(pd.read_parquet(a["ctx"].run_dir / "batches.parquet"),
                                  pd.read_parquet(b["ctx"].run_dir / "batches.parquet"))


def test_base_minimal_equals_lambda0(pair):
    a, m = pair["lam0"], pair["minimal"]
    assert m["queues"] is None
    for va, vm in zip([*a["encoder"].trainable_variables, a["P"]],
                      [*m["encoder"].trainable_variables, m["P"]]):
        np.testing.assert_allclose(va.numpy(), vm.numpy(), atol=1e-6)
    steps = pd.read_parquet(m["ctx"].run_dir / "training_steps.parquet")
    assert steps["n_c_min"].isna().all() and steps["mean_age_steps"].isna().all()
