"""Task 6.1: comparison methods (spec 8-8.3, M-D18/19/22).

Graph-side run stops (SAMPLER_ERROR, DISTANCE_RANGE_VIOLATION) are tf.debugging.Assert failures whose
message starts with the failure type: eagerly they raise InvalidArgumentError, and the training loop
maps them to the failure type via metric.train._failure (asserted here).
"""
import numpy as np
import pytest

from tests.metric.smoke.fixtures import make_sop_like, smoke_params

pytestmark = [pytest.mark.unit, pytest.mark.tf,
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]

D = 128


def _unit(a):
    a = np.asarray(a, np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _assert_stop(e, ftype):
    from metric.train import _failure
    assert ftype in e.value.message
    assert _failure(e.value, None, 0, None)[0] == ftype


def _np_d2(z):
    return np.clip(2 - 2 * z @ z.T, 0, 4)


def test_contrastive_matches_numpy(tf_or_skip):
    from metric.losses import contrastive_loss
    z = _unit(np.random.default_rng(0).standard_normal((8, 5)))
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    d2 = _np_d2(z)
    iu = np.triu_indices(8, 1)
    same = (y[:, None] == y[None, :])[iu]
    assert same.sum() == 4 and (~same).sum() == 24
    pos = d2[iu][same].mean()
    neg = (np.maximum(0, 1 - np.sqrt(d2[iu][~same])) ** 2).mean()
    got = contrastive_loss(tf_or_skip.constant(z, tf_or_skip.float32), y, 1.0, 1e-6)
    np.testing.assert_allclose(float(got), 0.5 * (pos + neg), rtol=1e-5)


def test_contrastive_zero_distance_no_nan(tf_or_skip):
    tf = tf_or_skip
    from metric.losses import contrastive_loss
    u, v, w = np.eye(4)[:3]
    h = tf.Variable(np.stack([u, u, v, w]).astype(np.float32))
    y = np.array([0, 1, 0, 1], np.int32)
    with tf.GradientTape() as t:
        L = contrastive_loss(tf.math.l2_normalize(h, 1), y, 1.0, 1e-6)
    g = t.gradient(L, h).numpy()
    assert np.isfinite(float(L)) and np.isfinite(g).all()
    # the D²=0 negative pair contributes m² = 1 to the P− mean (4 pairs)
    d2 = _np_d2(np.stack([u, u, v, w]))
    neg = [1.0] + [(max(0, 1 - np.sqrt(d2[i, j]))) ** 2 for i, j in ((0, 3), (1, 2), (2, 3))]
    pos = (d2[0, 2] + d2[1, 3]) / 2
    np.testing.assert_allclose(float(L), 0.5 * (pos + np.mean(neg)), rtol=1e-5)


def test_contrastive_empty_group(tf_or_skip):
    tf = tf_or_skip
    from metric.losses import contrastive_loss
    z = tf.constant(_unit(np.random.default_rng(1).standard_normal((4, 5))), tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError) as e:
        contrastive_loss(z, np.zeros(4, np.int32), 1.0, 1e-6)
    _assert_stop(e, "SAMPLER_ERROR")


def test_contrastive_distance_range(tf_or_skip):
    tf = tf_or_skip
    from metric.losses import contrastive_loss
    z = tf.constant(np.full((4, 3), 2.0), tf.float32)  # not normalised -> D² < -tol
    with pytest.raises(tf.errors.InvalidArgumentError) as e:
        contrastive_loss(z, np.array([0, 0, 1, 1], np.int32), 1.0, 1e-6)
    _assert_stop(e, "DISTANCE_RANGE_VIOLATION")


def test_triplet_batch_hard(tf_or_skip):
    from metric.losses import triplet_loss
    ang = np.array([0.0, 0.3, 1.0, 0.1, 2.0, 2.5])
    z = np.stack([np.cos(ang), np.sin(ang)], 1)
    y = np.array([0, 0, 0, 1, 1, 2], np.int32)  # class 2 anchor has no positive
    d2 = _np_d2(z)
    ls = []
    for a in range(5):
        p = [j for j in range(6) if y[j] == y[a] and j != a]
        n = [j for j in range(6) if y[j] != y[a]]
        ls.append(max(0, d2[a, p].max() - d2[a, n].min() + 0.2))
    got = triplet_loss(tf_or_skip.constant(z, tf_or_skip.float32), y, 0.2)
    np.testing.assert_allclose(float(got), np.mean(ls), rtol=1e-5)


def test_triplet_empty_anchor_set(tf_or_skip):
    tf = tf_or_skip
    from metric.losses import triplet_loss
    z = tf.constant(_unit(np.random.default_rng(2).standard_normal((4, 5))), tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError) as e:
        triplet_loss(z, np.arange(4, dtype=np.int32), 0.2)
    _assert_stop(e, "SAMPLER_ERROR")


def test_classifier_search_uses_normalized_h(tf_or_skip):
    tf = tf_or_skip
    from expcore.seeding import stream
    from metric.methods import build_method
    m = build_method("classifier", smoke_params(), 10, 3)
    assert m.trainable and not m.uses_queues and m.search_metric == "cosine"
    h = tf.constant(np.random.default_rng(0).standard_normal((4, D)), tf.float32)
    z = m.embed_for_search(h).numpy()
    assert z.shape == (4, D)
    np.testing.assert_allclose(z, _unit(h.numpy()), rtol=1e-5)
    kernel, bias = m.extra_variables
    assert kernel.shape == (D, 10) and bias.shape == (10,) and not bias.numpy().any()
    seed = int(stream(3, "init", 2).integers(2**31 - 1))
    ref = tf.keras.initializers.GlorotUniform(seed=seed)((D, 10))
    np.testing.assert_array_equal(kernel.numpy(), ref.numpy())
    L, _ = m.loss(h, np.arange(4, dtype=np.int32))
    logits = h.numpy() @ kernel.numpy()
    ref_l = np.mean(np.log(np.exp(logits).sum(1)) - logits[np.arange(4), np.arange(4)])
    np.testing.assert_allclose(float(L), ref_l, rtol=1e-4)


def test_fixed_features_dims(tf_or_skip):
    from metric.methods import build_method
    p = smoke_params()  # weights=None keeps the unit test offline; real runs use imagenet
    m = build_method("imagenet_fixed", p, 10, 0)
    assert not m.trainable and not m.needs_raw_pixels
    f = m.features(np.random.default_rng(0).uniform(-100, 100, (2, 56, 56, 3)).astype("float32"))
    assert f.shape == (2, 2048)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1, rtol=1e-5)

    r = build_method("rgb_hist", p, 10, 0)
    assert not r.trainable and r.needs_raw_pixels and r.search_metric == "cosine"
    img = np.array([[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [40, 100, 200]]], np.float32)
    h = r.features(img[None])
    assert h.shape == (1, 512)
    ref = np.zeros(512)
    ref[0], ref[511], ref[1 * 64 + 3 * 8 + 6] = 2, 1, 1
    np.testing.assert_allclose(h[0], ref / np.linalg.norm(ref), rtol=1e-6)


def test_noL2_raw_metric(tf_or_skip):
    tf = tf_or_skip
    from expcore.seeding import stream
    from metric.methods import build_method
    from metric.proxies import init_proxies
    m = build_method("proxy_softmax_noL2", smoke_params(), 10, 3)
    assert m.search_metric == "raw_l2" and not m.uses_queues
    np.testing.assert_array_equal(m.P.numpy(), init_proxies(10, D, stream(3, "init", 1)).numpy())
    h = tf.constant(np.random.default_rng(0).standard_normal((4, D)) * 3, tf.float32)
    np.testing.assert_array_equal(m.embed_for_search(h).numpy(), h.numpy())
    L, _ = m.loss(h, np.arange(4, dtype=np.int32))
    logits = h.numpy() @ m.P.numpy().T  # no κ, no normalisation
    ref = np.mean(np.log(np.exp(logits).sum(1)) - logits[np.arange(4), np.arange(4)])
    np.testing.assert_allclose(float(L), ref, rtol=1e-4)


def test_proxy_methods_share_init():
    from metric.methods import build_method
    a = build_method("proxy_cal", smoke_params(), 10, 3)
    b = build_method("base_minimal", smoke_params(), 10, 3)
    np.testing.assert_array_equal(a.P.numpy(), b.P.numpy())
    assert a.uses_queues and not b.uses_queues
    assert not build_method("proxy_cal", smoke_params(queues=False), 10, 3).uses_queues


class _FakeEncoder:
    def __init__(self, seed=0):
        import tensorflow as tf
        self.W = tf.Variable(np.random.default_rng(seed).standard_normal((3, D)), dtype=tf.float32)
        self.variables = self.trainable_variables = [self.W]

    def embed_raw(self, x):
        import tensorflow as tf
        return tf.reduce_mean(x, [1, 2]) @ self.W


def _step(method_id, n_cls=4):
    from metric.methods import build_method
    from metric.proxies import make_optimizer
    from metric.train_step import StepConfig, make_method_step
    m = build_method(method_id, smoke_params(), n_cls, 0)
    enc = _FakeEncoder()
    fn = make_method_step(enc, m, make_optimizer(1e-2), StepConfig(16, 0, 4, 1e-12, 1e-12))
    return m, enc, fn


def _x(n, seed=0):
    return np.random.default_rng(seed).uniform(-100, 100, (n, 4, 4, 3)).astype("float32")


@pytest.mark.parametrize("method_id", ["contrastive", "triplet", "classifier", "proxy_softmax_noL2"])
def test_method_step_updates_all(tf_or_skip, method_id):
    tf = tf_or_skip
    m, enc, fn = _step(method_id)
    before = [v.numpy().copy() for v in enc.trainable_variables + m.extra_variables]
    out = fn(_x(8), np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32), tf.constant(0, tf.int64))
    assert set(out) == {"L_base", "L_dev", "L_total", "n_cval"}
    assert float(out["L_dev"]) == 0 and int(out["n_cval"]) == 0
    assert float(out["L_total"]) == float(out["L_base"]) and np.isfinite(float(out["L_total"]))
    after = [v.numpy() for v in enc.trainable_variables + m.extra_variables]
    assert all(np.any(a != b) for a, b in zip(after, before))


def test_method_step_graph_errors(tf_or_skip):
    tf = tf_or_skip
    _, enc, fn = _step("contrastive")
    w = enc.W.numpy().copy()
    with pytest.raises(tf.errors.InvalidArgumentError) as e:
        fn(_x(4), np.zeros(4, np.int32), tf.constant(0, tf.int64))
    _assert_stop(e, "SAMPLER_ERROR")
    np.testing.assert_array_equal(enc.W.numpy(), w)  # no update on a failed step
    enc.W.assign(tf.zeros_like(enc.W))
    with pytest.raises(tf.errors.InvalidArgumentError) as e:
        fn(_x(4), np.array([0, 0, 1, 1], np.int32), tf.constant(1, tf.int64))
    _assert_stop(e, "INVALID_EMBEDDING_NORM")


def test_embedder_trainable(tf_or_skip):
    from metric.methods import build_method, make_embedder
    p = smoke_params()
    m, enc = build_method("triplet", p, 4, 0), _FakeEncoder()
    e = make_embedder(m, enc, p)
    z = e.embed_images(_x(5))
    assert z.shape == (5, D) and isinstance(z, np.ndarray)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, rtol=1e-5)
    h0 = e.weights_hash()
    assert e.method_id == "triplet" and h0 == make_embedder(m, enc, p).weights_hash()
    enc.W.assign_add(enc.W)
    assert e.weights_hash() != h0
    raw = make_embedder(build_method("proxy_softmax_noL2", p, 4, 0), enc, p).embed_images(_x(2))
    np.testing.assert_allclose(raw, enc.embed_raw(_x(2)).numpy(), rtol=1e-6)


def test_embedder_rgb_hist_gets_raw_pixels(tmp_path):
    from metric.embed import embed_split
    from metric.methods import build_method, make_embedder
    from metric.pipeline import _crop
    images, store = make_sop_like(n_classes=2, n_val_classes=2)
    p = smoke_params()
    m = build_method("rgb_hist", p, 2, 0)
    es = embed_split(make_embedder(m, None, p), images, "validation", store, p, "none", 0,
                     tmp_path, "mh")
    first = sorted(images.loc[images.split == "validation", "image_id"])[0]
    ref = m.features(_crop(store, first, 64, 56)[None].astype(np.float32))
    assert es.embeddings.shape[1] == 512 and es.method_id == "rgb_hist"
    np.testing.assert_allclose(es.embeddings[0], ref[0], rtol=1e-6)


def test_untrainable_rejected_by_run_training(tmp_path):
    from expcore.errors import ConfigError
    from metric.train import TrainContext, run_training
    images, store = make_sop_like(n_classes=2, n_val_classes=2)
    for mid in ("imagenet_fixed", "rgb_hist"):
        with pytest.raises(ConfigError):
            run_training(TrainContext(smoke_params(), 0, tmp_path / mid, images, store, method=mid))


def test_unknown_method():
    from expcore.errors import ConfigError
    from metric.methods import build_method
    with pytest.raises(ConfigError):
        build_method("nope", smoke_params(), 2, 0)
