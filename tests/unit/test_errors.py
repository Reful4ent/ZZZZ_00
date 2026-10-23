import pytest

from expcore.errors import (
    ConfigError,
    CountMismatchError,
    DataValidationError,
    InsufficientDataError,
    LeakageError,
    MixedRelationError,
    NotImplementedAlgorithm,
    ProtocolViolationError,
    ResultTamperedError,
    TechnicalError,
    TestAccessError,
    ThresholdLeakageError,
)


@pytest.mark.unit
def test_exit_codes():
    assert [e.exit_code for e in (ConfigError, DataValidationError, TechnicalError,
            ProtocolViolationError, NotImplementedAlgorithm)] == [2, 3, 4, 5, 6]
    for sub in (InsufficientDataError, CountMismatchError, LeakageError):
        assert issubclass(sub, DataValidationError)
    for sub in (TestAccessError, ThresholdLeakageError, ResultTamperedError, MixedRelationError):
        assert issubclass(sub, ProtocolViolationError)
