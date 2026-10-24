"""Task 6.1 smoke: a non-proxy method through run_training, and its head in the full checkpoint."""
import numpy as np
import pandas as pd
import pytest

from tests.metric.smoke.fixtures import make_sop_like, smoke_params

pytestmark = [pytest.mark.smoke, pytest.mark.tf,
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    pytest.importorskip("tensorflow")
    import metric.train as train
    images, store = make_sop_like()
    cap, real = {}, train.make_method_step

    def spy(encoder, method, optimizer, cfg):
        cap.update(encoder=encoder, method=method, opt=optimizer)
        return real(encoder, method, optimizer, cfg)

    ctx = train.TrainContext(params=smoke_params(epochs=1, steps_per_epoch=2), seed=7,
                             run_dir=tmp_path_factory.mktemp("clf"), images=images, store=store,
                             method="classifier")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "make_method_step", spy)
        cap["result"] = train.run_training(ctx)
    cap["dir"] = ctx.run_dir
    return cap


def test_classifier_run(run):
    d = run["dir"]
    assert run["result"]["status"] == "completed" and run["result"]["total_steps"] == 2
    ep = pd.read_parquet(d / "epochs.parquet")
    assert list(ep["epoch"]) == [1] and ep["L_dev_mean"].eq(0).all()
    steps = pd.read_parquet(d / "training_steps.parquet")
    assert steps["n_cval"].isna().all() and steps["n_c_min"].isna().all()
    assert steps["p_norm_mean"].isna().all() and (steps["L_total"] == steps["L_base"]).all()
    assert not (d / "queue_state" / "epoch-001.parquet").exists()
    assert int(run["opt"].iterations) == 2


def test_checkpoint_roundtrip_head(run):
    import tensorflow as tf
    from metric.checkpoint import restore_full
    r = tf.train.load_checkpoint(str(run["dir"] / "checkpoints/last/ckpt"))
    keys = r.get_variable_to_shape_map()
    assert any(k.startswith("head/") for k in keys) and not any(k.startswith("P/") for k in keys)
    head = run["method"].extra_variables
    saved = [v.numpy().copy() for v in head]
    for v in head:
        v.assign(tf.zeros_like(v))
    st = restore_full(run["dir"] / "checkpoints/last", run["encoder"], None, None, run["opt"],
                      head=head)
    assert st["global_step"] == 1
    for v, s in zip(head, saved):
        np.testing.assert_array_equal(v.numpy(), s)
