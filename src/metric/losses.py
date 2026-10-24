"""Proxy-loss building blocks (spec 6.1-6.3, 6.7). Imports TensorFlow."""
import tensorflow as tf


def l2_normalize_checked(x, eps, failure_type):
    """x / ||x||_2 row-wise; fails (no epsilon substitution) if norm <= eps or non-finite."""
    x = tf.convert_to_tensor(x)
    norm = tf.norm(x, axis=-1, keepdims=True)
    ok = tf.reduce_all(tf.math.is_finite(norm) & (norm > eps))
    with tf.control_dependencies(
        [tf.debugging.Assert(ok, [failure_type + ": row norm <= eps or non-finite", norm])]
    ):
        return x / tf.identity(norm)


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
