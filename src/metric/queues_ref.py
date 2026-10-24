"""Reference per-class FIFO queue (NumPy, no TensorFlow)."""
from collections import deque

import numpy as np


class QueueRef:
    def __init__(self, n_cls: int, K: int, d: int) -> None:
        self.n_cls, self.K, self.d = n_cls, K, d
        self._q = [deque(maxlen=K) for _ in range(n_cls)]

    def push(self, z: np.ndarray, labels: np.ndarray, step: int) -> None:
        for vec, c in zip(z, labels):  # row order; deque drops the oldest on overflow
            self._q[int(c)].append((np.array(vec, dtype=np.float32), step))

    @property
    def n(self) -> np.ndarray:
        return np.array([len(q) for q in self._q], dtype=np.int64)

    @property
    def S(self) -> np.ndarray:
        s = np.zeros((self.n_cls, self.d), dtype=np.float64)
        for c, q in enumerate(self._q):
            for vec, _ in q:
                s[c] += vec.astype(np.float64)
        return s

    def entries(self, c: int) -> list[tuple[np.ndarray, int]]:
        return list(self._q[c])

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        return (self.S / np.maximum(n, 1)[:, None]).astype(np.float32), n

    def mean_age(self, step: int) -> tuple[float | None, np.ndarray]:
        per = np.array([np.mean([step - t for _, t in q]) if q else np.nan for q in self._q])
        ages = [step - t for q in self._q for _, t in q]
        return (float(np.mean(ages)) if ages else None), per
