import json

import numpy as np
import pandas as pd
import pytest

from expcore.errors import DataValidationError, ProtocolViolationError, ThresholdLeakageError
from expcore.thresholds import Threshold, load_threshold, save_threshold, select_threshold

pytestmark = pytest.mark.unit

KW = dict(run_id="r", method_id="m", criterion_id="median", manifest_hash="a" * 64)
median = lambda s, y: float(np.median(s))  # noqa: E731


def _val(split="validation", direction="higher_is_match"):
    return pd.DataFrame({"split": [split] * 4, "y_match": [1, 1, 0, 0], "score_raw": [0.9, 0.7, 0.2, None],
                         "score_direction": [direction] * 4, "outcome": ["decided"] * 3 + ["reject"]})


def test_rejects_non_validation():   # INV-10
    df = pd.concat([_val(), _val("test")], ignore_index=True)
    with pytest.raises(ThresholdLeakageError):
        select_threshold(df, median, **KW)
    with pytest.raises(ThresholdLeakageError):
        select_threshold(_val("dev"), median, **KW)


def test_uses_unified_score():
    seen = {}

    def crit(s, y):
        seen["s"], seen["y"] = s.tolist(), y.tolist()
        return 0.0
    select_threshold(_val(direction="lower_is_match"), crit, **KW)
    assert seen == {"s": [-0.9, -0.7, -0.2], "y": [1, 1, 0]}   # null score (reject) is not passed


def test_only_validation_rows_pass():
    t = select_threshold(_val(), median, **KW)
    assert t.split_used == "validation" and t.value == 0.7 and t.n_validation_pairs == 4
    assert (t.run_id, t.method_id, t.criterion_id, t.manifest_hash) == ("r", "m", "median", "a" * 64)


def test_non_finite_threshold_rejected():
    with pytest.raises(DataValidationError):
        select_threshold(_val(), lambda s, y: float("nan"), **KW)


def test_roundtrip(tmp_path):
    t = select_threshold(_val(), median, **KW)
    save_threshold(t, tmp_path / "thresholds.json")
    assert load_threshold(tmp_path / "thresholds.json") == t
    assert "created_at" in json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))


def test_load_missing_or_wrong_split(tmp_path):
    with pytest.raises(ProtocolViolationError):
        load_threshold(tmp_path / "nope.json")
    t = Threshold(**KW, value=0.5, split_used="test", n_validation_pairs=4)
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps({**t.__dict__, "created_at": "2026-10-24T00:00:00Z"}), encoding="utf-8")
    with pytest.raises(ProtocolViolationError):
        load_threshold(p)


def test_corrupt_inputs_rejected():   # review P4-I1
    for col, value in (("y_match", None), ("y_match", 2), ("score_direction", "lower-is-match"),
                       ("score_raw", float("inf")), ("outcome", "error")):
        df = _val(); df.loc[0, col] = value
        with pytest.raises(DataValidationError):
            select_threshold(df, median, **KW)
    with pytest.raises(DataValidationError):
        select_threshold(_val().drop(columns=["outcome"]), median, **KW)


def test_only_scored_decisions_reach_criterion():   # review P4-I2: not_applicable and rejects are excluded
    seen = {}

    def crit(s, y):
        seen["s"] = s.tolist()
        return 0.0
    df = _val()
    df["score_raw"] = [0.9, 0.7, 0.2, 0.1]
    df["outcome"] = ["decided", "not_applicable", "decided", "reject"]
    select_threshold(df, crit, **KW)
    assert seen["s"] == [0.9, 0.2]


def test_save_is_idempotent(tmp_path):   # review P4-I3: resume / re-aggregate must not fail
    from expcore.errors import ResultTamperedError
    t = select_threshold(_val(), median, **KW)
    save_threshold(t, tmp_path / "thresholds.json")
    save_threshold(t, tmp_path / "thresholds.json")   # same content: no-op
    with pytest.raises(ResultTamperedError):
        save_threshold(Threshold(**{**t.__dict__, "value": 0.1}), tmp_path / "thresholds.json")
