"""HPatches split (spec §4.3): per prefix i_/v_, first ceil(20%·n) shuffled → validation, rest → test."""

from typing import Sequence

from expcore.errors import DataValidationError, InsufficientDataError
from expcore.splits.rng import ceil_pct, shuffled

PREFIXES = ("i_", "v_")
VALIDATION_PCT = 20


def split_hpatches(sequence_names: Sequence[str]) -> dict[str, list[str]]:
    bad = [n for n in sequence_names if not n.startswith(PREFIXES)]
    if bad:
        raise DataValidationError(f"HPatches split: sequence names without i_/v_ prefix: {sorted(bad)[:5]}")
    parts: dict[str, list[str]] = {"validation": [], "test": []}
    for prefix in PREFIXES:
        order = shuffled([n for n in sequence_names if n.startswith(prefix)])
        if not order:
            continue
        k = ceil_pct(len(order), VALIDATION_PCT)
        if k >= len(order):
            raise InsufficientDataError(f"HPatches split: prefix {prefix} with n={len(order)} leaves test empty")
        parts["validation"] += order[:k]
        parts["test"] += order[k:]
    return parts
