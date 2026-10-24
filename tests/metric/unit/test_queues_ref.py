import numpy as np
import pytest
from metric.queues_ref import QueueRef

pytestmark = pytest.mark.unit


def v(*x):
    return np.array(x, dtype=np.float32)


def test_partial_fill():
    q = QueueRef(2, 4, 2)
    q.push(np.stack([v(1, 2), v(3, 4)]), np.array([0, 0]), 0)
    assert q.n[0] == 2
    np.testing.assert_array_equal(q.S[0], [4, 6])
    e = q.entries(0)
    np.testing.assert_array_equal(e[0][0], v(1, 2))
    np.testing.assert_array_equal(e[1][0], v(3, 4))


def test_overflow_drops_oldest():
    q = QueueRef(1, 2, 2)
    for i, x in enumerate([v(1, 0), v(0, 1), v(2, 2)]):
        q.push(x[None], np.array([0]), i)
    e = q.entries(0)
    assert [t for _, t in e] == [1, 2]
    np.testing.assert_array_equal(q.S[0], [2, 3])


def test_absent_class_unchanged():
    q = QueueRef(2, 4, 2)
    q.push(v(1, 1)[None], np.array([0]), 0)
    q.push(v(5, 5)[None], np.array([1]), 1)
    assert q.n[0] == 1
    np.testing.assert_array_equal(q.S[0], [1, 1])


def test_more_new_than_K():
    q = QueueRef(1, 1, 2)
    q.push(np.stack([v(1, 1), v(2, 3)]), np.array([0, 0]), 0)
    e = q.entries(0)
    assert len(e) == 1
    np.testing.assert_array_equal(e[0][0], v(2, 3))
    np.testing.assert_array_equal(q.S[0], [2, 3])


def test_age():
    q = QueueRef(2, 4, 2)
    mean, per = q.mean_age(5)
    assert mean is None and np.isnan(per).all()
    q.push(v(1, 1)[None], np.array([0]), 3)
    mean, per = q.mean_age(5)
    assert mean == 2.0 and per[0] == 2.0 and np.isnan(per[1])


def test_centers():
    q = QueueRef(2, 4, 2)
    q.push(np.stack([v(1, 2), v(3, 4)]), np.array([0, 0]), 0)
    c, n = q.centers()
    np.testing.assert_allclose(c[0], [2, 3])
    assert c.dtype == np.float32 and (c[1] == 0).all() and list(n) == [2, 0]
