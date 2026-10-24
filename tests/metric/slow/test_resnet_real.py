"""Real ResNet50 (ImageNet weights, 224 px): checksum, encoder shape, one training step. Run with `-m slow`."""
import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.tf, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _params_sha():
    import yaml
    from metric.expconfigs import REPO
    t = yaml.safe_load((REPO / "configs/templates/metric.yaml").read_text(encoding="utf-8"))
    return t["params"]["model"]["weights_sha256"]


def test_backbone_checksum():
    pytest.importorskip("tensorflow")
    from expcore.primitives import run_check
    from metric.model import build_encoder, weights_file_sha256
    sha = weights_file_sha256()
    if sha is None:
        pytest.skip("ImageNet ResNet50 weights not cached")
    assert sha == _params_sha()
    run_check("PRIM-BACKBONE")  # expcore's registered check: output (None, 2048), 23 587 712 parameters
    enc = build_encoder("imagenet", 224, 128, dense_seed=0)
    assert enc.embed_raw(np.zeros((2, 224, 224, 3), np.float32)).shape == (2, 128)


def test_real_step_224():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from metric.model import build_encoder, weights_file_sha256
    from metric.train_step import StepConfig, make_proxy_step
    if weights_file_sha256() is None:
        pytest.skip("ImageNet ResNet50 weights not cached")
    enc = build_encoder("imagenet", 224, 128, dense_seed=0)
    rng = np.random.default_rng(0)
    P = tf.Variable(rng.standard_normal((8, 128)).astype(np.float32))
    P.assign(P / tf.norm(P, axis=1, keepdims=True))
    opt = tf.keras.optimizers.Adam(1e-5)
    step = make_proxy_step(enc, P, None, opt, StepConfig(kappa=16.0, lam=0.0, n_warm=1, eps_z=1e-12, eps_p=1e-6))
    bn = [v for v in enc.backbone.variables if "moving" in v.path]
    assert bn
    before = [v.numpy().copy() for v in bn]
    x = rng.uniform(-100, 100, (32, 224, 224, 3)).astype(np.float32)
    labels = np.repeat(np.arange(8), 4).astype(np.int32)
    out = step(x, labels, tf.constant(0, tf.int64))
    assert all(np.isfinite(float(out[k])) for k in ("L_base", "L_dev", "L_total"))
    assert np.allclose(np.linalg.norm(enc.embed(x[:8], 1e-12).numpy(), axis=1), 1.0, atol=1e-5)
    for b, a in zip(before, bn):
        np.testing.assert_array_equal(b, a.numpy())
