import numpy as np
import pandas as pd
import pytest

from expcore.errors import ConfigError, ProtocolViolationError
from expcore.io import dump_json
from expcore.schemas import validate_predictions
from metric.embed import EmbeddingSet
from metric.evaluate import (criterion_max_recall_at_fpr, evaluate_pairs, evaluate_retrieval,
                             predictions_rows)

pytestmark = pytest.mark.unit

PARAMS = {"search": {"block_q": 4, "block_g": 4, "k": 10, "d2_tol": 1e-6},
          "threshold": {"criterion_id": "max_recall_at_fpr_le_0.05", "fpr_max": 0.05}}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def es(ids, z, split="validation"):
    return EmbeddingSet(unit(z), np.array(ids), "wh", split, "clean", "m")


def ids_for(tmp_path):
    return {"out_dir": tmp_path, "run_id": "r", "method_id": "m", "manifest_hash": "h"}


def make_pairs():
    # a,b same; c,d same; cross pairs differ
    z = [[1, 0, 0], [1, .1, 0], [0, 1, 0], [0, 1, .1], [0, 0, 1]]
    ids = ["a", "b", "c", "d", "e"]
    pairs = pd.DataFrame({"pair_id": ["p0", "p1", "p2", "p3"], "query_id": ["a", "c", "a", "b"],
                          "reference_id": ["b", "d", "c", "e"], "y_match": np.array([1, 1, 0, 0], np.int8)})
    return es(ids, z), pairs


def test_threshold_fpr_constraint():
    d = np.array([.1, .2, .3, .4, .5, .6])
    y = np.array([1, 1, 0, 1, 0, 0])
    thr = criterion_max_recall_at_fpr(-d, y, fpr_max=0.5)  # FPR<=.5 allows d<=.5: recall 1 at d=.4 and .5
    assert thr == -0.4  # tie on recall -> larger s = smaller tD
    pred = -d >= thr
    assert (pred & (y == 0)).sum() / (y == 0).sum() <= 0.5
    assert pred[y == 1].mean() == 1.0
    assert criterion_max_recall_at_fpr(-np.array([.1, .2]), np.array([0, 1])) > -0.1  # nothing predicted


def test_threshold_test_readonly(tmp_path):
    e, pairs = make_pairs()
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(e, pairs, "test", PARAMS, tmp_path / "thresholds.json", ids=ids_for(tmp_path))
    bad = tmp_path / "t.json"
    dump_json({"run_id": "r", "method_id": "m", "criterion_id": "c", "value": -0.1, "split_used": "test",
               "n_validation_pairs": 1, "manifest_hash": "h", "created_at": "x"}, bad)
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(e, pairs, "test", PARAMS, bad, ids=ids_for(tmp_path))


def test_gallery_hash_checked(tmp_path):
    e, _ = make_pairs()
    classes = pd.Series([0, 0, 1, 1, 2], index=e.image_ids)
    evaluate_retrieval(e, e, classes, PARAMS, ids_for(tmp_path))
    other = EmbeddingSet(e.embeddings[:4], e.image_ids[:4], "wh", "validation", "clean", "m")
    with pytest.raises(ConfigError):
        evaluate_retrieval(other, other, classes, PARAMS, ids_for(tmp_path))


def run_all(tmp_path):
    e, pairs = make_pairs()
    th = tmp_path / "thresholds.json"
    sc = evaluate_pairs(e, pairs, "validation", PARAMS, th, ids=ids_for(tmp_path))
    from expcore.thresholds import load_threshold
    td = -load_threshold(th).value
    classes = pd.Series([0, 0, 1, 1, 2], index=e.image_ids)
    rt, summ = evaluate_retrieval(e, e, classes, PARAMS, ids_for(tmp_path), td=td)
    return e, pairs, sc, rt, summ, td


def test_predictions_contract(tmp_path):
    e, pairs, sc, rt, summ, td = run_all(tmp_path)
    assert (tmp_path / "pair_scores.parquet").exists() and (tmp_path / "retrieval.parquet").exists()
    assert summ["gallery_sha256"] and "y_hat" in rt
    pred = predictions_rows(sc, td, run_id="r", method_id="m", retrieval=rt)
    assert (pred.query_id == "e").any() and (pred.outcome == "not_applicable").sum() == 1  # class 2 has no positive
    ev = pd.concat([pairs, pd.DataFrame({"query_id": rt["query_id"]})], ignore_index=True)
    validate_predictions(pred[pred.pair_id.notna()], pairs, require_complete=True)
    validate_predictions(pred[pred.query_id.notna()], ev, require_complete=True)
    assert not pred.loc[pred.outcome == "decided", "score_raw"].isna().any()


def test_no_softmax_score(tmp_path):
    e, pairs, sc, rt, summ, td = run_all(tmp_path)
    pred = predictions_rows(sc, td, run_id="r", method_id="m")
    assert np.array_equal(pred["score_raw"].to_numpy(), sc["d"].to_numpy())
