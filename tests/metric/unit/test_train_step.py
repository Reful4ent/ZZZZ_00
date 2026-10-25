import numpy as np
import pytest

pytestmark = [pytest.mark.tf, pytest.mark.filterwarnings("ignore::DeprecationWarning")]

N_CLS, K, D, B = 6, 4, 16, 6
LAM = 0.7


def _setup(lam=LAM, with_queues=True, seed=0):
    import tensorflow as tf
    from metric.model import build_encoder
    from metric.proxies import init_proxies
    from metric.queues import ClassQueues
    from metric.train_step import StepConfig, make_proxy_step
    from metric.proxies import make_optimizer
    enc = build_encoder(None, 32, D, 3)
    P = init_proxies(N_CLS, D, np.random.default_rng(seed))
    q = ClassQueues(N_CLS, K, D) if with_queues else None
    opt = make_optimizer(1e-3)
    cfg = StepConfig(kappa=10.0, lam=lam, n_warm=1, eps_z=1e-8, eps_p=1e-8)
    step = make_proxy_step(enc, P, q, opt, cfg)
    return enc, P, q, opt, cfg, step


def _batch(seed=0, labels=(0, 1, 2, 0, 1, 2)):
    x = np.random.default_rng(seed).uniform(0, 255, (len(labels), 32, 32, 3)).astype("float32")
    return x, np.array(labels, np.int32)


def test_grad_theta_independent_of_dev(tf_or_skip):   # PRIEM-M2 / G2-154: grad_theta L_total does not depend on lambda
    tf = tf_or_skip
    enc0, P0, q0, _, _, step0 = _setup(lam=0.0)
    enc1, P1, q1, _, _, step1 = _setup(lam=1.0)
    enc1.set_weights(enc0.get_weights())   # identical encoder / P / queues / fresh Adam, same batch
    P1.assign(P0)
    x, y = _batch()
    step0(x, y, 0)
    step1(x, y, 0)
    # one Adam step is a function of the gradient alone: equal encoder updates <=> equal encoder gradients
    for a, b in zip(enc0.trainable_variables, enc1.trainable_variables):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
    assert np.abs(P0.numpy() - P1.numpy()).max() > 1e-6   # while the dev term does reach P


def test_grad_p_matches_formula(tf_or_skip):
    tf = tf_or_skip
    from metric.losses import base_loss, dev_loss, proxy_logits
    enc, P, q, *_ = _setup()
    # class 5 in C_val but absent from the batch; class 3,4 outside C_val
    q.push(tf.constant(np.random.default_rng(1).standard_normal((3, D)), tf.float32),
           np.array([0, 5, 5], np.int32), 0)
    x, y = _batch(labels=(0, 1, 0, 1, 0, 1))
    mu, n = q.centers()
    mu, n = mu.numpy(), n.numpy()
    mask = n >= 1
    z = enc.embed(x, 1e-8)

    def grad(use_dev):
        with tf.GradientTape() as t:
            L = base_loss(proxy_logits(z, P, 10.0, 1e-8), y)
            if use_dev:
                L = L + LAM * dev_loss(P, tf.constant(mu), tf.constant(n), 1)[0]
        return t.gradient(L, P).numpy()

    diff = grad(True) - grad(False)
    expect = 2 * LAM * mask[:, None] / mask.sum() * (P.numpy() - mu)
    np.testing.assert_allclose(diff, expect, atol=1e-5)
    assert np.abs(diff[5]).max() > 0
    assert not np.any(diff[~mask])


def test_p_equals_mu_zero_dev(tf_or_skip):
    from metric.losses import dev_loss
    enc, P, q, *_ = _setup()
    x, y = _batch()
    q.push(enc.embed(x, 1e-8), y, 0)
    mu, n = q.centers()
    P.assign(mu)
    assert float(dev_loss(P, mu, n, 1)[0]) == 0.0


def test_single_adam_update(tf_or_skip):
    enc, P, q, opt, cfg, step = _setup()
    x, y = _batch()
    out = step(x, y, 0)
    assert int(opt.iterations) == 1
    assert set(out) == {"L_base", "L_dev", "L_total", "n_cval"}
    assert int(out["n_cval"]) == 3  # classes 0,1,2 after the push


def test_base_minimal_no_queues(tf_or_skip):
    enc, P, q, opt, cfg, step = _setup(with_queues=False)
    x, y = _batch()
    out = step(x, y, 0)
    assert float(out["L_dev"]) == 0.0 and int(out["n_cval"]) == 0
    assert float(out["L_total"]) == pytest.approx(float(out["L_base"]))


def test_p_not_projected(tf_or_skip):
    enc, P, q, opt, cfg, step = _setup(lam=1.0)
    x, y = _batch()
    for i in range(3):
        step(x, y, i)
    assert np.abs(np.linalg.norm(P.numpy(), axis=1) - 1).max() > 1e-4


def _state(enc, P, q, opt):
    qs = [q.buf, q.n, q.start, q.S, q.ins_step]
    return [v.numpy().copy() for v in list(enc.variables) + [P] + qs] + [int(opt.iterations)]


def _assert_blocked(tf, setup, bad, msg):
    enc, P, q, opt, cfg, step = setup
    bad(enc, P)
    snap = _state(enc, P, q, opt)
    x, y = _batch()
    with pytest.raises(tf.errors.InvalidArgumentError, match=msg):
        step(x, y, 0)
    for a, b in zip(snap, _state(enc, P, q, opt)):
        np.testing.assert_array_equal(a, b)


def test_invalid_norm_blocks_update(tf_or_skip):
    tf = tf_or_skip
    _assert_blocked(tf, _setup(), lambda enc, P: enc.dense.kernel.assign(
        tf.zeros_like(enc.dense.kernel)), "INVALID_EMBEDDING_NORM")


def test_invalid_proxy_norm_blocks_update(tf_or_skip):
    tf = tf_or_skip

    def bad(enc, P):
        v = P.numpy()
        v[2] = 0.0
        P.assign(v)

    _assert_blocked(tf, _setup(), bad, "INVALID_PROXY_NORM")


def test_no_retrace_across_steps(tf_or_skip):
    enc, P, q, opt, cfg, step = _setup()
    x, y = _batch()
    for i in range(3):
        step(x, y, i)
    assert step.experimental_get_tracing_count() == 1


def test_queue_uses_theta_t(tf_or_skip):
    enc, P, q, opt, cfg, step = _setup()
    x, y = _batch(labels=(0, 1, 2, 3, 4, 5))
    z = enc.embed(x, 1e-8).numpy()
    step(x, y, 0)
    for c in range(N_CLS):
        np.testing.assert_allclose(q.ordered_entries(c)[0], z[c], atol=1e-5)
