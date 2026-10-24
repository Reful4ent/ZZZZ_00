"""Comparison methods (spec 8-8.3, M-D18/19/22) and the embedder used by metric.embed. Imports TF.

`params` is the spec 11 dict; the method id is the `algorithms` key (M-D32), passed separately.
Search on `embed_for_search(h)` with `search_metric` ("cosine" for unit rows, "raw_l2" otherwise).
Untrainable methods expose `features(images)`; rgb_hist reads raw [0,255] RGB crops
(`needs_raw_pixels`, honoured by metric.embed.embed_split), the rest take preprocess_input'd arrays.
"""
import hashlib
import json

import numpy as np
import tensorflow as tf

from expcore.errors import ConfigError
from expcore.seeding import stream
from metric.losses import (base_loss, classifier_loss, contrastive_loss, l2_normalize_checked,
                           proxy_logits, triplet_loss)
from metric.model import weights_hash
from metric.proxies import init_proxies

UNTRAINABLE = ("imagenet_fixed", "rgb_hist")
PROXY_STEP = ("proxy_cal", "base_minimal")  # trained by train_step.make_proxy_step


class Method:
    trainable, uses_queues, needs_raw_pixels, search_metric = True, False, False, "cosine"

    def __init__(self, method_id, params):
        self.method_id, self.eps_z = method_id, params["embed"]["eps_z"]
        self.extra_variables = []

    def embed_for_search(self, h):
        return l2_normalize_checked(h, self.eps_z, "INVALID_EMBEDDING_NORM")

    def loss(self, h, labels):
        raise NotImplementedError

    def features(self, images):
        raise TypeError(f"{self.method_id} is trainable: embed through its encoder")


class ProxyMethod(Method):
    """proxy_cal / base_minimal (normalised, κ) and proxy_softmax_noL2 (h·pᵀ, raw L2 search)."""

    def __init__(self, method_id, params, n_cls, seed):
        super().__init__(method_id, params)
        self.P = init_proxies(n_cls, params["model"]["embedding_dim"], stream(seed, "init", 1))
        self.extra_variables = [self.P]
        self.kappa, self.eps_p = params["proxy"]["kappa"], params["proxy"]["eps_p"]
        self.normalized = method_id != "proxy_softmax_noL2"
        self.uses_queues = method_id == "proxy_cal" and bool(params["calibration"]["queues"])
        if not self.normalized:
            self.search_metric = "raw_l2"

    def loss(self, h, labels):
        if self.normalized:
            logits = proxy_logits(self.embed_for_search(h), self.P, self.kappa, self.eps_p)
        else:
            logits = tf.matmul(h, self.P, transpose_b=True)
        return base_loss(logits, labels), {}

    def embed_for_search(self, h):
        return super().embed_for_search(h) if self.normalized else tf.convert_to_tensor(h)


class PairMethod(Method):
    """contrastive (M-1, default m=1) / triplet (M-2, default margin 0.2) on normalised z."""

    def __init__(self, method_id, params):
        super().__init__(method_id, params)
        m = params["loss"]["margin"]
        self.margin = float(m if m is not None else (1.0 if method_id == "contrastive" else 0.2))
        self.tol = params["search"]["d2_tol"]

    def loss(self, h, labels):
        z = self.embed_for_search(h)
        fn = contrastive_loss if self.method_id == "contrastive" else triplet_loss
        return fn(z, labels, self.margin, self.tol), {}


class ClassifierMethod(Method):
    """h (no normalisation) -> Dense(Ncls, bias) -> softmax CE; search on normalize(h) (spec 8.3)."""

    def __init__(self, params, n_cls, seed):
        super().__init__("classifier", params)
        kseed = int(stream(seed, "init", 2).integers(2**31 - 1))
        self.head = tf.keras.layers.Dense(
            n_cls, use_bias=True, bias_initializer="zeros",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kseed))
        self.head.build((None, params["model"]["embedding_dim"]))
        self.extra_variables = [self.head.kernel, self.head.bias]

    def loss(self, h, labels):
        return classifier_loss(self.head(h), labels), {}


class ImagenetFixed(Method):
    """M-D19: ResNet50 GAP (d=2048), L2-normalised, no training."""
    trainable = False

    def __init__(self, params):
        super().__init__("imagenet_fixed", params)
        crop = params["input"]["crop"]
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False, pooling="avg", weights=params["model"]["weights"],
            input_shape=(crop, crop, 3))

    def features(self, images):
        return self.embed_for_search(self.backbone(images, training=False)).numpy()

    def features_hash(self):
        return weights_hash(self.backbone, self.method_id)


class RgbHist(Method):
    """M-D18: joint 8×8×8 RGB histogram of the raw centre crop, sum-normalised, then L2."""
    trainable, needs_raw_pixels = False, True

    def __init__(self, params):
        super().__init__("rgb_hist", params)

    def features(self, images):
        b = np.clip(np.floor(np.asarray(images, np.float64) / 32), 0, 7).astype(np.int64)
        idx = (b[..., 0] * 64 + b[..., 1] * 8 + b[..., 2]).reshape(len(b), -1)
        h = np.stack([np.bincount(r, minlength=512) for r in idx]).astype(np.float64)
        h /= h.sum(1, keepdims=True)
        return (h / np.linalg.norm(h, axis=1, keepdims=True)).astype(np.float32)

    def features_hash(self):
        return hashlib.sha256(b"rgb_hist:joint8x8x8:raw_rgb_0_255:sum1:l2").hexdigest()


def build_method(method_id: str, params: dict, n_cls: int, seed: int) -> Method:
    if method_id in ("proxy_cal", "base_minimal", "proxy_softmax_noL2"):
        return ProxyMethod(method_id, params, n_cls, seed)
    if method_id in ("contrastive", "triplet"):
        return PairMethod(method_id, params)
    if method_id == "classifier":
        return ClassifierMethod(params, n_cls, seed)
    if method_id == "imagenet_fixed":
        return ImagenetFixed(params)
    if method_id == "rgb_hist":
        return RgbHist(params)
    raise ConfigError(f"unknown method_id {method_id!r}")


class Embedder:
    """metric.embed consumer interface: embed_images(x) -> ndarray, method_id, weights_hash()."""

    def __init__(self, method, encoder, params):
        self.method, self.encoder = method, encoder
        self.method_id, self.needs_raw_pixels = method.method_id, method.needs_raw_pixels
        inp = params["input"]
        self._extra = json.dumps({"method_id": method.method_id, "resize_short": inp["resize_short"],
                                  "crop": inp["crop"]}, sort_keys=True)

    def embed_images(self, x):
        if not self.method.trainable:
            return self.method.features(x)
        h = self.encoder.embed_raw(tf.constant(x, tf.float32))  # backbone in inference mode
        return self.method.embed_for_search(h).numpy()

    def weights_hash(self):
        if not self.method.trainable:
            return self.method.features_hash()
        return weights_hash(self.encoder, self._extra)  # search ignores the classifier head


def make_embedder(method, encoder, params) -> Embedder:
    return Embedder(method, encoder, params)
