"""KADID-10k split (spec §4.2): shuffled originals, first n_validation → validation, rest → test."""

from typing import Sequence

from expcore.errors import InsufficientDataError
from expcore.splits.rng import shuffled


def split_kadid(original_ids: Sequence[str], n_validation: int) -> dict[str, list[str]]:
    order = shuffled(original_ids)
    parts = {"validation": order[:n_validation], "test": order[n_validation:]}
    empty = [k for k, v in parts.items() if not v]
    if empty:
        raise InsufficientDataError(f"KADID split: empty part(s) {empty} for {len(order)} originals")
    return parts
