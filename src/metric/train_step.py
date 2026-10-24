"""Train steps in the spec 7.1 order: one forward, norm checks, (push, centers,) one Adam update."""
from dataclasses import dataclass

import tensorflow as tf

from metric.losses import base_loss, check_norm, dev_loss, l2_normalize_checked, proxy_logits


@dataclass
class StepConfig:
    kappa: float
    lam: float
    n_warm: int
    eps_z: float
    eps_p: float


def make_proxy_step(encoder, P, queues, optimizer, cfg: StepConfig):
    optimizer.build(encoder.trainable_variables + [P])  # avoid a second trace on first apply
    @tf.function(jit_compile=False, input_signature=[
        tf.TensorSpec([None, None, None, 3], tf.float32),
        tf.TensorSpec([None], tf.int32),
        tf.TensorSpec([], tf.int64)])
    def step(x, labels, step_idx):
        labels = tf.cast(labels, tf.int32)
        with tf.GradientTape() as tape:
            h = encoder.embed_raw(x)
            z = l2_normalize_checked(h, cfg.eps_z, "INVALID_EMBEDDING_NORM")
            L_base = base_loss(proxy_logits(z, P, cfg.kappa, cfg.eps_p), labels)  # checks P
            if queues is not None:
                with tf.control_dependencies([L_base]):  # L_base carries both norm asserts
                    queues.push(tf.stop_gradient(z), labels, step_idx)
                mu, n = queues.centers()
                L_dev, n_val = dev_loss(P, mu, n, cfg.n_warm)
            else:
                L_dev, n_val = tf.constant(0.0), tf.constant(0, tf.int32)
            L_total = L_base + cfg.lam * L_dev
        vs = encoder.trainable_variables + [P]
        grads = tape.gradient(L_total, vs)
        with tf.control_dependencies([L_total, *grads]):  # norm asserts must finish first
            optimizer.apply_gradients(zip(grads, vs))
        return {"L_base": L_base, "L_dev": L_dev, "L_total": L_total, "n_cval": n_val}

    return step


def make_method_step(encoder, method, optimizer, cfg: StepConfig):
    """Non-proxy trainable methods: same signature/outputs as make_proxy_step, L_dev = 0, no queues."""
    vs = encoder.trainable_variables + method.extra_variables
    optimizer.build(vs)

    @tf.function(jit_compile=False, input_signature=[
        tf.TensorSpec([None, None, None, 3], tf.float32),
        tf.TensorSpec([None], tf.int32),
        tf.TensorSpec([], tf.int64)])
    def step(x, labels, step_idx):
        with tf.GradientTape() as tape:
            h, _ = check_norm(encoder.embed_raw(x), cfg.eps_z, "INVALID_EMBEDDING_NORM")
            L, _ = method.loss(h, labels)
        grads = tape.gradient(L, vs)
        with tf.control_dependencies([L, *grads]):  # norm/sampler/D² asserts must finish first
            optimizer.apply_gradients(zip(grads, vs))
        return {"L_base": L, "L_dev": tf.constant(0.0), "L_total": L,
                "n_cval": tf.constant(0, tf.int32)}

    return step
