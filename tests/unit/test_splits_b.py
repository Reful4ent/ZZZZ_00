import pytest

from expcore.errors import InsufficientDataError
from expcore.splits.mvtec import split_mvtec_category
from expcore.splits.sop import split_sop

pytestmark = pytest.mark.unit

PARTS = ("dev", "validation", "synthetic_test")


def test_mvtec_boundaries():
    parts = split_mvtec_category([f"carpet/{i:03d}" for i in range(280)])
    assert [len(parts[k]) for k in PARTS] == [70, 70, 140]


def test_mvtec_odd_n():
    parts = split_mvtec_category([f"c/{i:03d}" for i in range(247)])
    assert [len(parts[k]) for k in PARTS] == [61, 61, 125]


def test_mvtec_too_small():
    with pytest.raises(InsufficientDataError):
        split_mvtec_category(["a", "b", "c"])


def test_sop_80_20():
    parts = split_sop(list(range(1, 11319)))
    assert len(parts["train"]) == (4 * 11318) // 5 and len(parts["train"]) + len(parts["validation"]) == 11318


def test_sop_numeric_sort():
    assert split_sop([2, 10, 1, 3, 4]) == split_sop([1, 2, 3, 4, 10])
    assert len(split_sop([2, 10, 1, 3, 4])["train"]) == 4


def test_sop_too_small():
    with pytest.raises(InsufficientDataError):
        split_sop([1])


def test_parts_disjoint_and_complete():
    for n in (4, 7, 100, 247):
        parts = split_mvtec_category([f"c/{i:03d}" for i in range(n)])
        assert sorted(sum(parts.values(), [])) == sorted(f"c/{i:03d}" for i in range(n))
    parts = split_sop(list(range(1, 51)))
    assert sorted(parts["train"] + parts["validation"]) == list(range(1, 51))
