import pytest

from expcore.errors import DataValidationError, InsufficientDataError
from expcore.splits.hpatches import split_hpatches
from expcore.splits.kadid import split_kadid
from expcore.splits.rng import ceil_pct, shuffled

pytestmark = pytest.mark.unit

# T-SPL-07: recorded once from PCG64(2026); a change means the NumPy stream changed.
REFERENCE_PERM_8 = ['a', 'b', 'f', 'g', 'd', 'h', 'e', 'c']


def test_ceil_pct():
    assert [ceil_pct(n, 20) for n in (1, 5, 57, 59)] == [1, 1, 12, 12]


def test_shuffled_independent_of_input_order():
    assert shuffled(["c", "a", "b"]) == shuffled(["b", "c", "a"])


def test_shuffled_reference_permutation():
    assert shuffled(list("abcdefgh")) == REFERENCE_PERM_8


def test_shuffled_ints_sorted_numerically():
    assert shuffled([10, 2, 1]) == shuffled([1, 2, 10]) and sorted(shuffled([10, 2, 1])) == [1, 2, 10]


def test_kadid_81_20_61():
    parts = split_kadid([f"I{n:02d}" for n in range(1, 82)], 20)
    assert (len(parts["validation"]), len(parts["test"])) == (20, 61) and not set(parts["validation"]) & set(parts["test"])


def test_kadid_empty_part():
    with pytest.raises(InsufficientDataError):
        split_kadid(["I01"], 1)


def test_hpatches_per_prefix_and_ceil():
    names = [f"i_{k:02d}" for k in range(57)] + [f"v_{k:02d}" for k in range(59)]
    parts = split_hpatches(names)
    assert sum(n.startswith("i_") for n in parts["validation"]) == 12
    assert sum(n.startswith("v_") for n in parts["validation"]) == 12
    assert sorted(parts["validation"] + parts["test"]) == sorted(names)


def test_hpatches_single_sequence_prefix_fails():
    with pytest.raises(InsufficientDataError):
        split_hpatches(["i_a", "v_a", "v_b", "v_c", "v_d", "v_e"])


def test_hpatches_unprefixed_name_fails():
    with pytest.raises(DataValidationError):
        split_hpatches(["x_a", "i_a", "i_b"])


def test_split_deterministic_across_calls():
    ids = [f"I{n:02d}" for n in range(1, 82)]
    assert split_kadid(ids, 20) == split_kadid(list(reversed(ids)), 20)
