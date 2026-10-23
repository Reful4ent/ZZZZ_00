"""Protocol vocabularies: relations, decision outcomes, reason codes and algorithms."""

from enum import Enum

from expcore.errors import NotImplementedAlgorithm

RELATIONS = ("P1_SAME_SCENE", "P2_NO_CHANGE_IN_REGION", "P3_SAME_PRODUCT")


class Outcome(str, Enum):
    DECIDED = "decided"
    REJECT = "reject"
    NOT_APPLICABLE = "not_applicable"


# Codes with these prefixes belong to exactly that outcome; unprefixed codes
# (e.g. SCORE_UNDEFINED) annotate decided rows.
_PREFIXES = {"REJECT_": Outcome.REJECT, "NOT_APPLICABLE_": Outcome.NOT_APPLICABLE}

_REASON_CODES: dict[str, Outcome] = {}
_ALGORITHMS: dict[tuple[str, str], object] = {}


def register_reason_code(code: str, outcome: Outcome) -> None:
    outcome = Outcome(outcome)
    prefixed = next((o for p, o in _PREFIXES.items() if code.startswith(p)), Outcome.DECIDED)
    if prefixed is not outcome:
        raise ValueError(f"reason code {code!r} cannot belong to outcome {outcome.value!r}")
    if _REASON_CODES.get(code, outcome) is not outcome:
        raise ValueError(f"reason code {code!r} already registered for {_REASON_CODES[code].value!r}")
    _REASON_CODES[code] = outcome


def is_registered(code: str, outcome: Outcome) -> bool:
    return _REASON_CODES.get(code) is Outcome(outcome)


def register_algorithm(module: str, method_id: str, predictor: object) -> None:
    key = (module, method_id)
    if _ALGORITHMS.get(key, predictor) is not predictor:
        raise ValueError(f"algorithm {module}/{method_id} already registered")
    _ALGORITHMS[key] = predictor


def get_algorithm(module: str, method_id: str) -> object:
    try:
        return _ALGORITHMS[(module, method_id)]
    except KeyError:
        raise NotImplementedAlgorithm(f"algorithm {module}/{method_id} is not registered") from None


register_reason_code("REJECT_GENERIC", Outcome.REJECT)
register_reason_code("NOT_APPLICABLE_NO_REGION", Outcome.NOT_APPLICABLE)
register_reason_code("SCORE_UNDEFINED", Outcome.DECIDED)
