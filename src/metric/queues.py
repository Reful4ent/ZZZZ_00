"""Per-class FIFO ring buffers on TensorFlow (state in non-trainable Variables)."""
import numpy as np
import tensorflow as tf

DRIFT_BLOCK = 1024   # classes per host transfer in max_sum_drift


class ClassQueues(tf.Module):
    def __init__(self, n_cls: int, K: int, d: int) -> None:
        super().__init__()
        self.n_cls, self.K, self.d = n_cls, K, d
        v = lambda x: tf.Variable(x, trainable=False)  # noqa: E731
        self.buf = v(tf.zeros((n_cls, K, d), tf.float32))
        self.start = v(tf.zeros((n_cls,), tf.int32))
        self.n = v(tf.zeros((n_cls,), tf.int32))
        self.S = v(tf.zeros((n_cls, d), tf.float64))
        self.ins_step = v(tf.zeros((n_cls, K), tf.int64))

    def push(self, z, labels, step) -> None:
        K = self.K
        labels = tf.cast(labels, tf.int32)
        B = tf.shape(labels)[0]
        same = tf.cast(labels[:, None] == labels[None, :], tf.int32)  # [j, j']
        rank = tf.reduce_sum(tf.linalg.band_part(same, -1, 0), 1) - 1  # j' < j (same class)
        m = tf.reduce_sum(same, 1)
        keep = rank >= m - K
        r = rank - tf.maximum(m - K, 0)
        n_j, start_j = tf.gather(self.n, labels), tf.gather(self.start, labels)
        slot = (start_j + n_j + r) % K
        evict = keep & (n_j + r >= K)
        idx = tf.stack([labels, slot], 1)
        old = tf.gather_nd(self.buf, idx)  # read before scatter
        z = tf.cast(z, tf.float32)
        kf = tf.cast(keep, tf.float64)[:, None]
        ef = tf.cast(evict, tf.float64)[:, None]
        delta = tf.math.unsorted_segment_sum(
            kf * tf.cast(z, tf.float64) - ef * tf.cast(old, tf.float64), labels, self.n_cls)
        kidx = tf.boolean_mask(idx, keep)
        # in place: tensor_scatter_nd_update + assign would copy the whole buffer every step
        self.buf.scatter_nd_update(kidx, tf.boolean_mask(z, keep))
        steps = tf.fill([tf.shape(kidx)[0]], tf.cast(step, tf.int64))
        self.ins_step.scatter_nd_update(kidx, steps)
        self.S.assign_add(delta)
        cnt = tf.math.unsorted_segment_sum(tf.ones_like(labels), labels, self.n_cls)
        w = tf.minimum(cnt, K)
        overflow = tf.maximum(self.n + w - K, 0)
        self.start.assign((self.start + overflow) % K)
        self.n.assign(tf.minimum(self.n + w, K))

    def centers(self):
        n = tf.identity(self.n)
        mu = self.S / tf.cast(tf.maximum(n, 1), tf.float64)[:, None]
        return tf.cast(mu, tf.float32), n

    def _valid(self) -> np.ndarray:
        pos = (np.arange(self.K)[None, :] - self.start.numpy()[:, None]) % self.K
        return pos < self.n.numpy()[:, None]

    def age_stats(self, step) -> dict:
        age = int(step) - self.ins_step.numpy()
        valid = self._valid()
        n = valid.sum(1)
        per = np.where(n > 0, (age * valid).sum(1) / np.maximum(n, 1), np.nan)
        return {"mean_age": float(age[valid].mean()) if valid.any() else None, "per_class_age": per}

    def max_sum_drift(self, block: int = DRIFT_BLOCK) -> float:
        valid, S, drift = self._valid(), self.S.numpy(), 0.0
        for c0 in range(0, self.n_cls, block):   # blockwise: a float64 copy of all of buf is ~2x M_Q on host
            s = (self.buf[c0:c0 + block].numpy().astype(np.float64) * valid[c0:c0 + block, :, None]).sum(1)
            drift = max(drift, float(np.abs(S[c0:c0 + block] - s).max()))
        return drift

    def ordered_entries(self, c: int) -> np.ndarray:
        n, start = int(self.n[c]), int(self.start[c])
        return self.buf[c].numpy()[(start + np.arange(n)) % self.K]
