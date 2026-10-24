import pytest
from metric.memory import memory_breakdown, queue_bytes

pytestmark = pytest.mark.unit


def test_queue_bytes():
    assert queue_bytes(9054, 32, 128) == 9054 * 32 * 128 * 4


def test_breakdown_sums():
    b = memory_breakdown(10, 4, 8, 1000)
    assert b["M_Q"] == 10 * 4 * 8 * 4
    assert b["sums"] == 10 * 8 * 8
    assert b["pointers"] == 2 * 10 * 4
    assert b["ins_step"] == 10 * 4 * 8
    assert b["proxies"] == 10 * 8 * 4
    assert b["weights"] == 4000
    assert b["adam"] == 2 * (1000 + 80) * 4
    assert b["total_static"] == sum(v for k, v in b.items() if k != "total_static")
