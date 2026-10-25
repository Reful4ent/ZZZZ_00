"""Общие типы методов сравнения."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OK = "OK"
ZERO_ENERGY = "ZERO_ENERGY"
ZERO_VARIANCE = "ZERO_VARIANCE"
SIZE_MISMATCH = "SIZE_MISMATCH"
UNDEFINED_STATUSES = (ZERO_ENERGY, ZERO_VARIANCE, SIZE_MISMATCH)

DISTANCE = "distance"
SIMILARITY = "similarity"

# Порог undefined-ветвей NCC/ZNCC (ФОТО-04, Ф-2, Ф-3).
UNDEFINED_EPS = 1e-12


@dataclass(frozen=True)
class Result:
    """Исходная оценка пары: value (D или S) либо undefined с кодом статуса."""

    value: float
    status: str = OK

    @property
    def defined(self) -> bool:
        return self.status == OK

    @staticmethod
    def undefined(status: str) -> "Result":
        if status not in UNDEFINED_STATUSES:
            raise ValueError(f"unknown undefined status {status!r}")
        return Result(math.nan, status)


@dataclass
class Representation:
    method_id: str
    config_hash: str
    data: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)
