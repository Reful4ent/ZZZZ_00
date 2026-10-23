"""MVTec AD split per category (spec §4.4): N//4 → dev, N//4 → validation, rest → synthetic_test."""

from typing import Sequence

from expcore.errors import InsufficientDataError
from expcore.splits.rng import shuffled


def split_mvtec_category(train_good_keys: Sequence[str]) -> dict[str, list[str]]:
    order = shuffled(train_good_keys)
    q = len(order) // 4
    if q == 0:
        raise InsufficientDataError(f"MVTec split: N={len(order)} < 4 normal train images leaves a part empty")
    return {"dev": order[:q], "validation": order[q:2 * q], "synthetic_test": order[2 * q:]}
