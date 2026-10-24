"""Proxy variable init and optimizer (spec 5.2, 7)."""
import numpy as np
import tensorflow as tf


def init_proxies(n_cls: int, d: int, rng: np.random.Generator) -> tf.Variable:
    p = rng.standard_normal((n_cls, d))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return tf.Variable(p.astype(np.float32), trainable=True, name="proxies")


def make_optimizer(lr: float):
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
