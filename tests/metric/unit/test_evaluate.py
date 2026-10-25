import numpy as np
import pandas as pd
import pytest

from expcore.errors import ConfigError, ProtocolViolationError
from expcore.io import dump_json
from expcore.schemas import validate_predictions
from expcore.thresholds import load_threshold
from metric.embed import EmbeddingSet
from metric.evaluate import (criterion_max_recall_at_fpr, evaluate_pairs, evaluate_retrieval,
                             pair_prediction_rows, query_prediction_rows)

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
    z = [[1, 0, 0], [1, .1, 0], [0, 1, 0], [0, 1, .1], [0, 0, 1]]
    pairs = pd.DataFrame({"pair_id": ["p0", "p1", "p2", "p3"], "query_id": ["a", "c", "a", "b"],
                          "reference_id": ["b", "d", "c", "e"], "y_match": np.array([1, 1, 0, 0], np.int8),
                          "split": "validation"})
    return es(["a", "b", "c", "d", "e"], z), pairs


def test_threshold_fpr_constraint():
    d, y = np.array([.1, .2, .2, .3]), np.array([1, 1, 0, 0])
    assert criterion_max_recall_at_fpr(-d, y, fpr_max=0.5) == -0.2  # FPR .5 ok, recall 1
    assert criterion_max_recall_at_fpr(-d, y, fpr_max=0.4) == -0.1  # tied boundary infeasible
    # equal recall at several feasible thresholds -> larger unified thr (smaller tD)
    assert criterion_max_recall_at_fpr(-np.array([.1, .2, .3, .4]), np.array([1, 0, 0, 0]), 1.0) == -0.1
    assert criterion_max_recall_at_fpr(-np.array([.1, .2, .3]), np.array([1, 1, 0]), 1.0) == -0.2
    # negatives score highest: nothing feasible -> predict nothing
    thr = criterion_max_recall_at_fpr(-np.array([.1, .2]), np.array([0, 1]))
    assert thr == np.nextafter(-0.1, np.inf)


def test_threshold_test_readonly(tmp_path):
    e, pairs = make_pairs()
    et, pt = EmbeddingSet(e.embeddings, e.image_ids, "wh", "test", "clean", "m"), pairs.assign(split="test")
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(et, pt, "test", PARAMS, tmp_path / "thresholds.json", ids=ids_for(tmp_path))
    bad = tmp_path / "t.json"
    dump_json({"run_id": "r", "method_id": "m", "criterion_id": "c", "value": -0.1, "split_used": "test",
               "n_validation_pairs": 1, "manifest_hash": "h", "created_at": "x"}, bad)
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(et, pt, "test", PARAMS, bad, ids=ids_for(tmp_path))


def test_split_guard(tmp_path):
    e, pairs = make_pairs()
    t = EmbeddingSet(e.embeddings, e.image_ids, "wh", "test", "clean", "m")
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(t, pairs, "validation", PARAMS, tmp_path / "th.json", ids=ids_for(tmp_path))
    with pytest.raises(ProtocolViolationError):
        evaluate_pairs(e, pairs.assign(split="test"), "validation", PARAMS, tmp_path / "th.json",
                       ids=ids_for(tmp_path))
    assert not list(tmp_path.iterdir())


def test_gallery_hash_checked(tmp_path):
    e, _ = make_pairs()
    classes = pd.Series([0, 0, 1, 1, 2], index=e.image_ids)
    evaluate_retrieval(e, e, classes, PARAMS, ids_for(tmp_path), expected_gallery_ids=list(e.image_ids))
    other = EmbeddingSet(e.embeddings[:4], e.image_ids[:4], "wh", "validation", "clean", "m")
    with pytest.raises(ConfigError):
        evaluate_retrieval(other, other, classes, PARAMS, ids_for(tmp_path), expected_gallery_ids=list(e.image_ids))


def run_all(tmp_path):
    e, pairs = make_pairs()
    th = tmp_path / "thresholds.json"
    sc = evaluate_pairs(e, pairs, "validation", PARAMS, th, ids=ids_for(tmp_path))
    td = -load_threshold(th).value
    classes = pd.Series([0, 0, 1, 1, 2], index=e.image_ids)
    rt, summ = evaluate_retrieval(e, e, classes, PARAMS, ids_for(tmp_path),
                                  expected_gallery_ids=list(e.image_ids), td=td)
    return e, pairs, sc, rt, summ, td


def test_predictions_contract(tmp_path):
    e, pairs, sc, rt, summ, td = run_all(tmp_path)
    cols = pd.read_parquet(tmp_path / "pair_scores.parquet").columns.tolist()
    assert cols == ["run_id", "method_id", "pair_id", "split", "query_id", "reference_id", "s_cos", "d2", "d"]
    assert summ["gallery_sha256"] and rt.loc[~rt["valid"], "y_hat"].isna().all()
    validate_predictions(pair_prediction_rows(sc, td, run_id="r", method_id="m"), pairs, require_complete=True)
    qp = query_prediction_rows(rt, td, run_id="r", method_id="m")
    assert (qp.outcome == "not_applicable").sum() == 1  # class 2 has no positive
    validate_predictions(qp, pd.DataFrame({"query_id": rt["query_id"]}), require_complete=True)
    assert not qp.loc[qp.outcome == "decided", "score_raw"].isna().any()


def test_no_softmax_score(tmp_path):
    e, pairs, sc, rt, summ, td = run_all(tmp_path)
    pred = pair_prediction_rows(sc, td, run_id="r", method_id="m")
    assert np.array_equal(pred["score_raw"].to_numpy(), sc["d"].to_numpy())
