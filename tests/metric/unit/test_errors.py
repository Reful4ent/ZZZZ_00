import pytest
from metric.errors import RunStop

pytestmark = pytest.mark.unit


def test_runstop_carries_type():
    e = RunStop("OOM", {"step": 3})
    assert e.failure_type == "OOM" and e.details["step"] == 3 and "OOM" in str(e) and e.exit_code == 4
