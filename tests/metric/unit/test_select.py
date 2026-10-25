import pytest
from metric.select import Candidate, RunResult, SelectError, select_best

pytestmark = pytest.mark.unit

SEEDS = [11, 23, 37]


def cand(order, sums, n=100, status="completed", tag="c"):
    runs = [RunResult(f"{tag}{order}-{s}", s, status, v, n, 5) for s, v in zip(SEEDS, sums)]
    return Candidate({"lam": order}, order, runs)


def test_select_max_mean():
    best, table = select_best([cand(0, (10, 10, 10)), cand(1, (10, 10, 11))], SEEDS)
    assert best.order == 1
    assert table[0]["sum_correct_at_1"] == 30 and table[1]["sum_correct_at_1"] == 31
    assert table[1]["mean_val_recall_at_1"] == pytest.approx(31 / 300)
    assert table[1]["runs"][0] == {"run_id": "c1-11", "seed": 11, "correct_at_1": 10, "n_valid": 100, "best_epoch": 5}


def test_select_tie_smaller_order():
    best, _ = select_best([cand(1, (10, 10, 10)), cand(0, (9, 10, 11))], SEEDS)
    assert best.order == 0


def test_select_missing_or_failed():
    c = cand(0, (1, 2, 3))
    c.runs.pop()
    with pytest.raises(SelectError, match=r"c0-11, c0-23.*do not match seeds"):
        select_best([c], SEEDS)
    bad = cand(1, (1, 2, 3))
    bad.runs[2] = RunResult("bad-37", 37, "failed", None, None, None)
    with pytest.raises(SelectError, match="bad-37"):
        select_best([bad], SEEDS)


def test_select_duplicate_seed():
    c = cand(0, (1, 2, 3))
    c.runs[2] = RunResult("dup", 11, "completed", 3, 100, 1)
    with pytest.raises(SelectError):
        select_best([c], SEEDS)


def test_select_mismatched_denominator():
    c = cand(1, (1, 2, 3))
    c.runs[0] = RunResult("odd", 11, "completed", 1, 99, 1)
    with pytest.raises(SelectError):
        select_best([cand(0, (1, 2, 3)), c], SEEDS)


@pytest.mark.parametrize("n_valid,correct", [(0, 0), (10, 11), (10, -1)])
def test_select_rejects_impossible_counts(n_valid, correct):
    c = cand(0, (1, 2, 3), n=10)
    c.runs[1] = RunResult("odd-23", 23, "completed", correct, n_valid, 1)
    with pytest.raises(SelectError, match="odd-23"):
        select_best([c], SEEDS)
