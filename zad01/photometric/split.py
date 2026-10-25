"""Разбиение оригиналов на validation/test по original_id (PCG64(split_seed))."""

from __future__ import annotations

import numpy as np

PARTS = ("validation", "test")


def split_originals(original_ids, seed: int, fractions: dict) -> dict[str, str]:
    """Возвращает {original_id: часть}. Детерминировано: порядок входа не влияет на результат."""
    ids = sorted(set(original_ids))
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(ids))
    n_val = int(round(len(ids) * fractions["validation"]))
    n_val = min(max(n_val, 1), len(ids) - 1)
    return {ids[j]: ("validation" if rank < n_val else "test") for rank, j in enumerate(order)}
