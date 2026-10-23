"""Protocol exception hierarchy; each top-level class carries its CLI exit code."""


class ProtocolError(Exception):
    exit_code: int = 1


class ConfigError(ProtocolError):
    exit_code = 2


class DataValidationError(ProtocolError):
    exit_code = 3


class InsufficientDataError(DataValidationError):
    pass


class CountMismatchError(DataValidationError):
    pass


class LeakageError(DataValidationError):
    pass


class TechnicalError(ProtocolError):
    exit_code = 4


class ProtocolViolationError(ProtocolError):
    exit_code = 5


class TestAccessError(ProtocolViolationError):
    __test__ = False  # not a pytest test class despite the name


class ThresholdLeakageError(ProtocolViolationError):
    pass


class ResultTamperedError(ProtocolViolationError):
    pass


class MixedRelationError(ProtocolViolationError):
    pass


class NotImplementedAlgorithm(ProtocolError):
    exit_code = 6
