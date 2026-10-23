"""SOP split of official train classes (spec §4.5): first (4·M)//5 shuffled class_ids → train, rest → validation."""

from typing import Sequence

from expcore.errors import InsufficientDataError
from expcore.splits.rng import shuffled


def split_sop(class_ids: Sequence[int]) -> dict[str, list[int]]:
    order = shuffled([int(c) for c in class_ids])
    k = (4 * len(order)) // 5
    parts = {"train": order[:k], "validation": order[k:]}
    empty = [p for p, v in parts.items() if not v]
    if empty:
        raise InsufficientDataError(f"SOP split: empty part(s) {empty} for M={len(order)} classes")
    return parts
