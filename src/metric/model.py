"""ResNet50 backbone (BN frozen in inference mode) + bias-free Dense head (spec 5.1)."""
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from metric.losses import l2_normalize_checked

_WEIGHTS_FILE = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


class Encoder(tf.keras.Model):
    def __init__(self, backbone, dense):
        super().__init__()
        self.backbone, self.dense = backbone, dense

    def embed_raw(self, x):
        return self.dense(self.backbone(x, training=False))

    def embed(self, x, eps_z):
        return l2_normalize_checked(self.embed_raw(x), eps_z, "INVALID_EMBEDDING_NORM")


def build_encoder(weights, input_size, embedding_dim, dense_seed) -> Encoder:
    backbone = tf.keras.applications.ResNet50(
        include_top=False, pooling="avg", weights=weights,
        input_shape=(input_size, input_size, 3))
    dense = tf.keras.layers.Dense(
        embedding_dim, use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=dense_seed))
    enc = Encoder(backbone, dense)
    enc.embed_raw(tf.zeros((1, input_size, input_size, 3)))  # build variables
    return enc


def weights_file_sha256():
    p = Path.home() / ".keras" / "models" / _WEIGHTS_FILE
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def weights_hash(model, extra: str) -> str:
    h = hashlib.sha256()
    for v in model.variables:  # architecture order; names are auto-generated, so not hashed
        a = np.ascontiguousarray(v.numpy())
        h.update(f"{a.shape}{a.dtype}".encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()
