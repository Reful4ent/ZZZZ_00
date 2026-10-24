import numpy as np
import pandas as pd
import pytest

from metric.steplog import STEP_COLUMNS, StepLogger, step_stats

pytestmark = pytest.mark.unit


def _row(step, epoch, stats):
    return {"run_id": "r", "global_step": step, "epoch": epoch, "step_in_epoch": step % 2,
            "lambda": 0.1, "L_base": 1.0, "L_dev": 0.0, "L_total": 1.0, "n_cval": 0,
            **stats, "step_time_ms": 5.0, "data_time_ms": 1.0}


def test_steplog_shards_and_null_age(tmp_path):
    p = np.eye(3, 4, dtype=np.float32)
    empty = step_stats(p, n=np.zeros(3, np.int32), mu=np.zeros((3, 4), np.float32), K=4,
                       mean_age=None)
    assert empty["mean_age_steps"] is None and empty["mu_norm_min"] is None
    assert empty["n_classes_empty"] == 3 and empty["p_norm_max"] == pytest.approx(1.0)
    filled = step_stats(p, n=np.array([4, 1, 0]), mu=np.full((3, 4), 0.5, np.float32), K=4,
                        mean_age=2.0)
    assert filled["mu_norm_min"] == pytest.approx(1.0) and filled["n_classes_full"] == 1
    none = step_stats(p)
    assert none["n_c_min"] is None and none["mean_age_steps"] is None

    log = StepLogger(tmp_path)
    log.log(_row(0, 1, empty))
    log.log(_row(1, 1, filled))
    log.end_epoch(1)
    log.log(_row(2, 2, none) | {"n_cval": None})
    log.end_epoch(2)
    log.finalize()
    parts = sorted((tmp_path / "training_steps").glob("part-*.parquet"))
    assert [x.name for x in parts] == ["part-001.parquet", "part-002.parquet"]
    df = pd.read_parquet(tmp_path / "training_steps.parquet")
    assert list(df.columns) == STEP_COLUMNS and list(df["global_step"]) == [0, 1, 2]
    assert df["mean_age_steps"].isna().tolist() == [True, False, True]
    assert df["mu_norm_mean"].isna().tolist() == [True, False, True]
    assert df["n_cval"].isna().tolist() == [False, False, True]
