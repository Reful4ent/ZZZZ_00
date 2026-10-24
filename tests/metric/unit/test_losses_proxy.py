import numpy as np
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture
def L(tf_or_skip):
    from metric import losses
    return losses


def _logsumexp(a):
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


def test_base_loss_matches_numpy(tf_or_skip, L, rng):
    tf = tf_or_skip
    z = rng.normal(size=(4, 3)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    P = rng.normal(size=(5, 3)).astype(np.float32)
    y = np.array([0, 2, 4, 2])
    kappa = 10.0
    Pv = tf.Variable(P)
    got = L.base_loss(L.proxy_logits(tf.constant(z), Pv, kappa, 1e-12), tf.constant(y))
    s = z.astype(np.float64) @ (P / np.linalg.norm(P, axis=1, keepdims=True)).astype(np.float64).T
    ks = kappa * s
    want = -np.mean(ks[np.arange(4), y] - _logsumexp(ks))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-6)
    np.testing.assert_array_equal(Pv.numpy(), P)  # P is not normalized in place


def test_dev_loss_matches_direct(tf_or_skip, L, rng):
    tf = tf_or_skip
    P = rng.normal(size=(4, 3)).astype(np.float32)
    mu = rng.normal(size=(4, 3)).astype(np.float32)
    n = np.array([0, 3, 5, 4])
    ld, nv = L.dev_loss(tf.Variable(P), tf.constant(mu), tf.constant(n), 4)
    want = np.mean([np.sum((P[c] - mu[c]) ** 2) for c in (2, 3)])
    np.testing.assert_allclose(ld.numpy(), want, rtol=1e-5)
    assert int(nv.numpy()) == 2 and nv.dtype == tf.int32


def test_dev_loss_empty_cval(tf_or_skip, L, rng):
    tf = tf_or_skip
    P = tf.Variable(rng.normal(size=(3, 2)).astype(np.float32))
    mu = tf.constant(rng.normal(size=(3, 2)).astype(np.float32))
    with tf.GradientTape() as tape:
        ld, nv = L.dev_loss(P, mu, tf.constant([0, 1, 2]), 4)
    g = tape.gradient(ld, P)
    assert float(ld.numpy()) == 0.0 and int(nv.numpy()) == 0
    assert np.all(np.isfinite(g.numpy()))


def test_zero_center_ok(tf_or_skip, L):
    tf = tf_or_skip
    P = tf.Variable(np.ones((2, 3), np.float32))
    ld, _ = L.dev_loss(P, tf.zeros((2, 3)), tf.constant([5, 5]), 4)
    assert np.isfinite(ld.numpy()) and float(ld.numpy()) == pytest.approx(3.0)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_invalid_norm_raises(tf_or_skip, L, bad):
    tf = tf_or_skip
    h = np.array([[1.0, 0.0], [bad, bad]], np.float32)
    with pytest.raises(tf.errors.InvalidArgumentError, match="INVALID_EMBEDDING_NORM"):
        L.l2_normalize_checked(tf.constant(h), 1e-12, "INVALID_EMBEDDING_NORM")
    with pytest.raises(tf.errors.InvalidArgumentError, match="INVALID_EMBEDDING_NORM"):
        tf.function(lambda x: L.l2_normalize_checked(x, 1e-12, "INVALID_EMBEDDING_NORM"), autograph=False)(tf.constant(h))


def test_invalid_proxy_norm(tf_or_skip, L):
    tf = tf_or_skip
    P = tf.Variable(np.zeros((2, 2), np.float32))
    with pytest.raises(tf.errors.InvalidArgumentError, match="INVALID_PROXY_NORM"):
        L.proxy_logits(tf.constant([[1.0, 0.0]]), P, 10.0, 1e-12)


def test_same_dtype(tf_or_skip, L, rng):
    tf = tf_or_skip
    z = L.l2_normalize_checked(tf.constant(rng.normal(size=(2, 3)).astype(np.float32)), 1e-12, "INVALID_EMBEDDING_NORM")
    p = L.l2_normalize_checked(tf.Variable(rng.normal(size=(2, 3)).astype(np.float32)), 1e-12, "INVALID_PROXY_NORM")
    assert z.dtype == p.dtype == tf.float32
