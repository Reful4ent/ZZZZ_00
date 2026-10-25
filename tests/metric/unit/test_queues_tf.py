import numpy as np
import pytest

pytestmark = pytest.mark.tf

N_CLS, B, D = 5, 8, 4


def _check(q, ref, step):
    np.testing.assert_array_equal(q.n.numpy(), ref.n)
    np.testing.assert_allclose(q.S.numpy(), ref.S, atol=1e-12)
    for c in range(N_CLS):
        e = ref.entries(c)
        got = q.ordered_entries(c)
        assert got.shape[0] == len(e)
        for i, (v, t) in enumerate(e):
            np.testing.assert_array_equal(got[i], v)
        if e:
            K = q.buf.shape[1]
            idx = (int(q.start[c]) + np.arange(len(e))) % K
            assert list(q.ins_step.numpy()[c, idx]) == [t for _, t in e]
    mu, n = q.centers()
    rmu, rn = ref.centers()
    np.testing.assert_array_equal(mu.numpy(), rmu)
    assert np.all(np.isfinite(mu.numpy()))
    a = q.age_stats(step)
    mean, per = a["mean_age"], a["per_class_age"]
    rmean, rper = ref.mean_age(step)
    assert (mean is None) == (rmean is None)
    if mean is not None:
        assert mean == pytest.approx(rmean)
    np.testing.assert_allclose(per, rper)


@pytest.mark.parametrize("K", [1, 2, 3, 8])
@pytest.mark.parametrize("compiled", [False, True])
def test_matches_reference_random(tf_or_skip, K, compiled):
    tf = tf_or_skip
    from metric.queues import ClassQueues
    from metric.queues_ref import QueueRef

    rng = np.random.default_rng(K)
    q, ref = ClassQueues(N_CLS, K, D), QueueRef(N_CLS, K, D)
    push = tf.function(q.push, autograph=False) if compiled else q.push
    for step in range(200):
        # restrict to few classes on some steps -> absent classes and m_c > K
        ncls = rng.integers(1, N_CLS + 1)
        labels = rng.integers(0, ncls, B).astype(np.int32)
        z = rng.normal(size=(B, D)).astype(np.float32)
        push(tf.constant(z), tf.constant(labels), tf.constant(step, tf.int64))
        ref.push(z, labels, step)
        _check(q, ref, step)
    assert q.max_sum_drift() < 1e-9


def test_centers_not_renormalized(tf_or_skip):
    tf = tf_or_skip
    from metric.queues import ClassQueues

    q = ClassQueues(2, 4, 2)
    q.push(tf.constant([[1, 0], [-1, 0]], tf.float32), tf.constant([0, 0]), tf.constant(0, tf.int64))
    assert np.linalg.norm(q.centers()[0].numpy()[0]) == 0
    q = ClassQueues(2, 4, 2)
    z = np.array([[1, 0], [0, 1], [-1, 0]], np.float32)
    q.push(tf.constant(z), tf.constant([0, 0, 0]), tf.constant(0, tf.int64))
    assert np.linalg.norm(q.centers()[0].numpy()[0]) < 1


def test_drift_detects_corruption(tf_or_skip):
    tf = tf_or_skip
    from metric.queues import ClassQueues

    q = ClassQueues(2, 3, 2)
    q.push(tf.ones((2, 2)), tf.constant([0, 1]), tf.constant(0, tf.int64))
    assert q.max_sum_drift() < 1e-9
    q.S.assign_add(tf.constant([[1e-3, 0], [0, 0]], tf.float64))
    assert q.max_sum_drift() > 1e-6


def test_push_not_differentiable(tf_or_skip):
    tf = tf_or_skip
    from metric.queues import ClassQueues

    q = ClassQueues(2, 3, 2)
    z = tf.ones((2, 2))
    with tf.GradientTape() as tape:
        tape.watch(z)
        q.push(tf.stop_gradient(z), tf.constant([0, 1]), tf.constant(0, tf.int64))
        loss = tf.reduce_sum(q.centers()[0])
    assert tape.gradient(loss, z) is None


def test_drift_blockwise_equals_full(tf_or_skip):
    tf = tf_or_skip
    from metric.queues import ClassQueues

    rng = np.random.default_rng(0)
    q = ClassQueues(N_CLS, 3, D)
    for step in range(5):
        q.push(tf.constant(rng.normal(size=(B, D)).astype(np.float32)),
               tf.constant(rng.integers(0, N_CLS, B).astype(np.int32)), tf.constant(step, tf.int64))
    q.S.assign_add(tf.constant(rng.normal(size=(N_CLS, D)) * 1e-3))
    full = np.abs(q.S.numpy() - (q.buf.numpy().astype(np.float64) * q._valid()[:, :, None]).sum(1)).max()
    for block in (1, 2, N_CLS, 1024):
        assert q.max_sum_drift(block=block) == full


def test_push_updates_in_place(tf_or_skip):
    """No TensorScatterUpdate: that op copies the whole buffer (~M_Q extra memory) every step."""
    tf = tf_or_skip
    from metric.queues import ClassQueues

    q = ClassQueues(N_CLS, 3, D)
    fn = tf.function(q.push, autograph=False).get_concrete_function(
        tf.TensorSpec([None, D], tf.float32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int64))
    ops = {op.type for op in fn.graph.get_operations()}
    assert "TensorScatterUpdate" not in ops and "ResourceScatterNdUpdate" in ops
