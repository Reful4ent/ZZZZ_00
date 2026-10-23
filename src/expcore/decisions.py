"""Decision model (spec §5.4, §5.7, §6.1): outcome, score direction and thresholding."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from expcore.errors import DataValidationError
from expcore.registry import Outcome, is_registered


class ScoreDirection(str, Enum):
    HIGHER_IS_MATCH = "higher_is_match"
    LOWER_IS_MATCH = "lower_is_match"


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    y_pred: int | None
    score_raw: float | None
    score_direction: ScoreDirection
    reason_code: str | None
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome) or not isinstance(self.score_direction, ScoreDirection):
            raise DataValidationError(f"INV-05: outcome/score_direction must be enum members: {self}")
        if self.outcome is Outcome.DECIDED:
            if isinstance(self.y_pred, bool) or self.y_pred not in (0, 1):
                raise DataValidationError(f"INV-05: decided requires y_pred in {{0,1}}, got {self.y_pred!r}")
        elif self.y_pred is not None:
            raise DataValidationError(f"INV-05: {self.outcome.value} requires y_pred=null, got {self.y_pred!r}")
        if self.outcome is not Outcome.DECIDED or self.reason_code is not None:
            if not is_registered(self.reason_code, self.outcome):
                raise DataValidationError(
                    f"INV-05: reason_code {self.reason_code!r} is not registered for {self.outcome.value}")
        if self.score_raw is None:
            if self.reason_code is None:
                raise DataValidationError("INV-06: score_raw=null requires a reason_code")
        elif not _finite(self.score_raw):
            raise DataValidationError(f"INV-06: score_raw must be finite, got {self.score_raw!r}")
        bad = {k: v for k, v in self.components.items() if not _finite(v)}
        if bad:
            raise DataValidationError(f"INV-06: components must be finite: {bad}")


def unified_score(score_raw: float | None, direction: ScoreDirection) -> float | None:
    """score_match: identity for higher_is_match, negate for lower_is_match."""
    if score_raw is None:
        return None
    return score_raw if ScoreDirection(direction) is ScoreDirection.HIGHER_IS_MATCH else -score_raw


def apply_threshold(score_raw: float, direction: ScoreDirection, threshold: float, tie_rule: str) -> int:
    """Threshold is on the unified (score_match) scale; tie_rule decides the boundary."""
    s = unified_score(score_raw, direction)
    if tie_rule == ">=":
        return int(s >= threshold)
    if tie_rule == ">":
        return int(s > threshold)
    raise ValueError(f"unknown tie_rule {tie_rule!r}")
