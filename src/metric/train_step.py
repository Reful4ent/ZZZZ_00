"""Proxy train step in the spec 7.1 order: one forward, push, centers, one Adam update."""
from dataclasses import dataclass

import tensorflow as tf

from metric.losses import base_loss, dev_loss, l2_normalize_checked, proxy_logits


@dataclass
class StepConfig:
    kappa: float
    lam: float
    n_warm: int
    eps_z: float
    eps_p: float


def make_proxy_step(encoder, P, queues, optimizer, cfg: StepConfig):
    @tf.function(jit_compile=False)
    def step(x, labels, step_idx):
        labels = tf.cast(labels, tf.int32)
        with tf.GradientTape() as tape:
            h = encoder.embed_raw(x)
            z = l2_normalize_checked(h, cfg.eps_z, "INVALID_EMBEDDING_NORM")
            L_base = base_loss(proxy_logits(z, P, cfg.kappa, cfg.eps_p), labels)  # checks P
            if queues is not None:
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
