"""Split RNG (D-02, D-03) and integer share boundaries (D-04)."""

from typing import Sequence, TypeVar

import numpy as np

K = TypeVar("K", str, int)

SPLIT_SEED = 2026


def shuffled(keys: Sequence[K], seed: int = SPLIT_SEED) -> list[K]:
    """Sort (code points for str, numerically for int), then permute with a fresh PCG64(seed)."""
    ordered = sorted(keys)
    rng = np.random.Generator(np.random.PCG64(seed))
    return [ordered[i] for i in rng.permutation(len(ordered))]


def ceil_pct(n: int, pct: int) -> int:
    """ceil(pct% of n) in integers only."""
    return (pct * n + 99) // 100
