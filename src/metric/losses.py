"""Loss building blocks: proxy (spec 6.1-6.3, 6.7) and baselines (8.1-8.3). Imports TensorFlow.

Graph-side run stops are tf.debugging.Assert failures whose message starts with the failure type;
metric.train maps the resulting InvalidArgumentError back to that type.
"""
import tensorflow as tf


def check_norm(x, eps, failure_type):
    """(x, row norms), both gated on an assert that every norm is finite and > eps."""
    x = tf.convert_to_tensor(x)
    norm = tf.norm(x, axis=-1, keepdims=True)
    ok = tf.reduce_all(tf.math.is_finite(norm) & (norm > eps))
    with tf.control_dependencies(
        [tf.debugging.Assert(ok, [failure_type + ": row norm <= eps or non-finite", norm])]
    ):
        return tf.identity(x), tf.identity(norm)


def l2_normalize_checked(x, eps, failure_type):
    """x / ||x||_2 row-wise; fails (no epsilon substitution) if norm <= eps or non-finite."""
    x, norm = check_norm(x, eps, failure_type)
    return x / norm


def proxy_logits(z, P, kappa, eps_p):
    """kappa * z @ normalize(P)^T; the normalized copy is used for the score only."""
    p_hat = l2_normalize_checked(tf.convert_to_tensor(P), eps_p, "INVALID_PROXY_NORM")
    return kappa * tf.matmul(z, p_hat, transpose_b=True)


def base_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )


def dev_loss(P, mu, n, n_warm):
    """(L_dev, n_val): mean ||p_c - sg(mu_c)||^2 over classes with n_c >= n_warm; 0 if none."""
    mask = tf.convert_to_tensor(n) >= n_warm
    n_val = tf.reduce_sum(tf.cast(mask, tf.int32))
    d2 = tf.reduce_sum(tf.square(tf.convert_to_tensor(P) - tf.stop_gradient(mu)), axis=-1)
    total = tf.reduce_sum(tf.where(mask, d2, tf.zeros_like(d2)))
    return total / tf.cast(tf.maximum(n_val, 1), d2.dtype), n_val


classifier_loss = base_loss  # softmax CE on Dense(Ncls) logits, no temperature (spec 8.3)


def _require(cond, failure_type, msg):
    return tf.debugging.Assert(cond, [failure_type + ": " + msg])


def cosine_d2(z, tol):
    """D² = 2 - 2 z zᵀ of normalised rows, checked against [-tol, 4+tol] and clipped (spec 9.2)."""
    d2 = 2.0 - 2.0 * tf.matmul(z, z, transpose_b=True)
    ok = tf.reduce_all(tf.math.is_finite(d2) & (d2 >= -tol) & (d2 <= 4.0 + tol))
    with tf.control_dependencies([_require(ok, "DISTANCE_RANGE_VIOLATION", "D2 outside tolerance")]):
        return tf.clip_by_value(d2, 0.0, 4.0)


def _same_class(labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    return tf.equal(labels[:, None], labels[None, :])


def contrastive_loss(z, labels, margin: float, tol: float):
    """M-1: half-sum of the P+ mean D² and the P− mean max(0, m - D)², unordered pairs."""
    d2 = cosine_d2(z, tol)
    upper = tf.linalg.band_part(tf.ones_like(d2), 0, -1) - tf.linalg.band_part(tf.ones_like(d2), 0, 0)
    upper = upper > 0
    same = _same_class(labels)
    pos, neg = same & upper, ~same & upper
    n_pos = tf.reduce_sum(tf.cast(pos, d2.dtype))
    n_neg = tf.reduce_sum(tf.cast(neg, d2.dtype))
    with tf.control_dependencies([_require((n_pos > 0) & (n_neg > 0), "SAMPLER_ERROR",
                                           "empty P+ or P- group")]):
        n_pos, n_neg = tf.identity(n_pos), tf.identity(n_neg)
    zero = tf.zeros_like(d2)
    d = tf.sqrt(tf.where(d2 > 0, d2, tf.ones_like(d2)))  # safe sqrt argument: no NaN gradient
    hinge = tf.where(d2 > 0, tf.square(tf.nn.relu(margin - d)), tf.fill(tf.shape(d2), margin**2))
    l_pos = tf.reduce_sum(tf.where(pos, d2, zero)) / n_pos
    l_neg = tf.reduce_sum(tf.where(neg, hinge, zero)) / n_neg
    return 0.5 * (l_pos + l_neg)


def triplet_loss(z, labels, margin: float, tol: float = 1e-6):
    """M-2 batch-hard on D²: farthest positive (other image), nearest negative, mean over valid anchors."""
    d2 = cosine_d2(z, tol)
    same = _same_class(labels)
    pos = same & ~tf.eye(tf.shape(d2)[0], dtype=tf.bool)
    neg = ~same
    hp = tf.reduce_max(tf.where(pos, d2, tf.fill(tf.shape(d2), -float("inf"))), axis=1)
    hn = tf.reduce_min(tf.where(neg, d2, tf.fill(tf.shape(d2), float("inf"))), axis=1)
    valid = tf.reduce_any(pos, 1) & tf.reduce_any(neg, 1)
    n_a = tf.reduce_sum(tf.cast(valid, d2.dtype))
    with tf.control_dependencies([_require(n_a > 0, "SAMPLER_ERROR", "empty anchor set")]):
        n_a = tf.identity(n_a)
    per = tf.nn.relu(hp - hn + margin)  # invalid anchors give relu(-inf) = 0 and are masked anyway
    return tf.reduce_sum(tf.where(valid, per, tf.zeros_like(per))) / n_a
