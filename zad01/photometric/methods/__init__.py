"""represent(image, method_config) и compare(query_repr, reference_repr) — API ТЗ 01, раздел 6."""

from __future__ import annotations

from photometric.io import check_rgb01
from photometric.methods.base import (
    DISTANCE,
    OK,
    SIMILARITY,
    SIZE_MISMATCH,
    UNDEFINED_STATUSES,
    ZERO_ENERGY,
    ZERO_VARIANCE,
    Representation,
    Result,
)
from photometric.methods.registry import DEFAULT_METHOD_IDS, METHODS, resolve

__all__ = [
    "represent",
    "compare",
    "Representation",
    "Result",
    "METHODS",
    "DEFAULT_METHOD_IDS",
    "resolve",
    "OK",
    "ZERO_ENERGY",
    "ZERO_VARIANCE",
    "SIZE_MISMATCH",
    "UNDEFINED_STATUSES",
    "DISTANCE",
    "SIMILARITY",
]


def represent(image, method_config) -> Representation:
    """Строит представление метода из общего RGB float64 [0,1]."""
    spec, params, chash = resolve(method_config)
    check_rgb01(image)
    data = spec.represent_fn(image, params)
    return Representation(spec.method_id, chash, data, {"params": params})


def compare(query_repr: Representation, reference_repr: Representation) -> Result:
    """Исходная оценка пары: D для расстояний, S для сходств, либо undefined."""
    if query_repr.method_id != reference_repr.method_id or query_repr.config_hash != reference_repr.config_hash:
        raise ValueError(
            f"representations of different methods/configs: "
            f"{query_repr.method_id}/{query_repr.config_hash} vs {reference_repr.method_id}/{reference_repr.config_hash}"
        )
    spec = METHODS[query_repr.method_id]
    params = query_repr.meta.get("params") or resolve(query_repr.method_id)[1]
    return spec.compare_fn(query_repr.data, reference_repr.data, params)
