"""P3 run-stop exception (exit code 4 via TechnicalError)."""
from expcore.errors import TechnicalError


class RunStop(TechnicalError):
    def __init__(self, failure_type: str, details: dict | None = None) -> None:
        super().__init__(failure_type)
        self.failure_type = failure_type
        self.details = details or {}
