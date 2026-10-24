import numpy as np
import pytest

pytestmark = pytest.mark.tf


def _enc(d=128):
    from metric.model import build_encoder
    return build_encoder(None, 32, d, 7)


def _x(n=4, seed=0):
    return np.random.default_rng(seed).uniform(0, 255, (n, 32, 32, 3)).astype("float32")


def test_encoder_structure(tf_or_skip):
    tf = tf_or_skip
    enc = _enc()
    assert isinstance(enc.dense, tf.keras.layers.Dense)
    assert enc.dense.units == 128 and enc.dense.use_bias is False and enc.dense.activation is tf.keras.activations.linear
    assert not any(isinstance(l, tf.keras.layers.Dropout) for l in enc.backbone.layers)
    assert enc.trainable_variables[-1] is enc.dense.kernel or enc.trainable_variables[-1].path == enc.dense.kernel.path
    z = enc.embed(_x(), 1e-8).numpy()
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_bn_stats_frozen(tf_or_skip):
    tf = tf_or_skip
    enc = _enc()
    x = _x()
    enc.embed_raw(x)
    bns = [l for l in enc.backbone.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    before = [(l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy(), l.gamma.numpy().copy()) for l in bns]
    opt = tf.keras.optimizers.Adam(1e-2)
    with tf.GradientTape() as t:
        loss = tf.reduce_sum(enc.embed_raw(x))
    vs = enc.trainable_variables
    opt.apply_gradients(zip(t.gradient(loss, vs), vs))
    for l, (m, v, _) in zip(bns, before):
        np.testing.assert_array_equal(l.moving_mean.numpy(), m)
        np.testing.assert_array_equal(l.moving_variance.numpy(), v)
    assert any(not np.array_equal(l.gamma.numpy(), g) for l, (_, _, g) in zip(bns, before))


def test_proxy_init(tf_or_skip):
    from metric.proxies import init_proxies
    P = init_proxies(10, 128, np.random.default_rng(1))
    np.testing.assert_allclose(np.linalg.norm(P.numpy(), axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(P.numpy(), init_proxies(10, 128, np.random.default_rng(1)).numpy())
    assert P.dtype.name == "float32" and P.trainable


def test_weights_hash_and_file_sha(tf_or_skip):
    from metric.model import weights_file_sha256, weights_hash
    enc = _enc(8)
    enc.embed_raw(_x(1))
    h = weights_hash(enc, "x")
    assert h == weights_hash(enc, "x") and h != weights_hash(enc, "y")
    enc.dense.kernel.assign(enc.dense.kernel + 1.0)
    assert weights_hash(enc, "x") != h
    s = weights_file_sha256()
    assert s is None or len(s) == 64


def test_weights_hash_name_independent(tf_or_skip):
    from metric.model import weights_hash
    a, b = _enc(8), _enc(8)  # b gets suffixed layer names
    b.set_weights(a.get_weights())  # backbone init is unseeded; make weights bit-identical
    assert weights_hash(a, "x") == weights_hash(b, "x")
    b.dense.kernel.assign(b.dense.kernel + 1.0)
    assert weights_hash(a, "x") != weights_hash(b, "x")
