import pytest

from expcore.decisions import Decision, ScoreDirection, apply_threshold, unified_score
from expcore.errors import DataValidationError, NotImplementedAlgorithm
from expcore.registry import (
    RELATIONS,
    Outcome,
    get_algorithm,
    is_registered,
    register_algorithm,
    register_reason_code,
)

pytestmark = pytest.mark.unit

DECIDED, REJECT, NOT_APPLICABLE = Outcome.DECIDED, Outcome.REJECT, Outcome.NOT_APPLICABLE
HIGHER_IS_MATCH, LOWER_IS_MATCH = ScoreDirection.HIGHER_IS_MATCH, ScoreDirection.LOWER_IS_MATCH


def test_unified_score():
    assert unified_score(0.3, LOWER_IS_MATCH) == -0.3 and unified_score(0.3, HIGHER_IS_MATCH) == 0.3 and unified_score(None, LOWER_IS_MATCH) is None


def test_tie_rule():
    assert apply_threshold(0.5, HIGHER_IS_MATCH, 0.5, ">=") == 1
    assert apply_threshold(0.5, HIGHER_IS_MATCH, 0.5, ">") == 0
    assert apply_threshold(0.2, LOWER_IS_MATCH, -0.3, ">=") == 1   # unified -0.2 >= -0.3


def test_zero_score_is_not_reject():   # ДАН-05
    d = Decision(DECIDED, 0, 0.0, HIGHER_IS_MATCH, None, {})
    assert d.outcome is DECIDED


def test_reject_needs_registered_code():
    with pytest.raises(DataValidationError):
        Decision(REJECT, None, None, HIGHER_IS_MATCH, "UNKNOWN", {})
    assert Decision(REJECT, None, None, HIGHER_IS_MATCH, "REJECT_GENERIC", {}).y_pred is None


def test_decided_requires_y_pred():
    with pytest.raises(DataValidationError):
        Decision(DECIDED, None, 0.1, HIGHER_IS_MATCH, None, {})
    with pytest.raises(DataValidationError):
        Decision(DECIDED, 2, 0.1, HIGHER_IS_MATCH, None, {})


def test_reject_and_not_applicable_shape():   # INV-05
    with pytest.raises(DataValidationError, match="INV-05"):
        Decision(REJECT, 1, None, HIGHER_IS_MATCH, "REJECT_GENERIC", {})
    with pytest.raises(DataValidationError, match="INV-05"):
        Decision(NOT_APPLICABLE, None, None, HIGHER_IS_MATCH, "REJECT_GENERIC", {})
    assert Decision(NOT_APPLICABLE, None, None, HIGHER_IS_MATCH, "NOT_APPLICABLE_NO_REGION", {}).outcome is NOT_APPLICABLE


def test_score_must_be_finite_or_null_with_reason():   # INV-06
    with pytest.raises(DataValidationError, match="INV-06"):
        Decision(DECIDED, 1, float("nan"), HIGHER_IS_MATCH, None, {})
    with pytest.raises(DataValidationError, match="INV-06"):
        Decision(DECIDED, 1, None, HIGHER_IS_MATCH, None, {})
    assert Decision(DECIDED, 1, None, HIGHER_IS_MATCH, "SCORE_UNDEFINED", {}).score_raw is None
    with pytest.raises(DataValidationError, match="INV-06"):
        Decision(DECIDED, 1, 0.5, HIGHER_IS_MATCH, None, {"c": float("inf")})


def test_reason_code_prefix_rules():
    assert is_registered("REJECT_GENERIC", REJECT) and not is_registered("REJECT_GENERIC", DECIDED)
    with pytest.raises(ValueError):
        register_reason_code("REJECT_X", NOT_APPLICABLE)
    with pytest.raises(ValueError):
        register_reason_code("NO_PREFIX", REJECT)
    register_reason_code("REJECT_TOO_FEW_MATCHES", REJECT)
    assert is_registered("REJECT_TOO_FEW_MATCHES", REJECT)


def test_relations():
    assert RELATIONS == ("P1_SAME_SCENE", "P2_NO_CHANGE_IN_REGION", "P3_SAME_PRODUCT")


def test_get_unregistered_algorithm():
    with pytest.raises(NotImplementedAlgorithm):
        get_algorithm("photometric", "nope")
    marker = object()
    register_algorithm("photometric", "test_method", marker)
    assert get_algorithm("photometric", "test_method") is marker
    with pytest.raises(ValueError):
        register_algorithm("photometric", "test_method", object())
