import pandas as pd
import pytest

from expcore.errors import DataValidationError, MixedRelationError
from expcore.metrics import compute_metrics, confusion_2x3

pytestmark = pytest.mark.unit

R = "P1_SAME_SCENE"


def _df(rows):
    return pd.DataFrame([dict(y_match=y, outcome=o, y_pred=p, relation_id=R) for y, o, p in rows])


# 10 pairs, N+=6, N-=4; accepted 8: TP=3, FN=1, FP=1, TN=3; 2 rejects among positives
HAND_DF = _df([(1, "decided", 1)] * 3 + [(1, "decided", 0)] + [(1, "reject", None)] * 2
              + [(0, "decided", 1)] + [(0, "decided", 0)] * 3)
ALL_REJECT = _df([(1, "reject", None)] * 3 + [(0, "reject", None)] * 2)


def test_hand_example():
    m = compute_metrics(HAND_DF)
    assert (m.tp, m.fn, m.fp, m.tn, m.rej_pos, m.rej_neg) == (3, 1, 1, 3, 2, 0)
    assert (m.n_pos, m.n_neg, m.n_not_applicable) == (6, 4, 0)
    assert m.coverage == 0.8 and m.rho_rej == pytest.approx(0.2)
    assert m.recall_all == 3/6 and m.fpr_all == 1/4 and m.correct_all == 6/10
    assert m.recall_cond == 3/4 and m.fpr_cond == 1/4 and m.correct_cond == 6/8
    assert m.coverage_pos == 4/6 and m.coverage_neg == 1.0
    assert m.positive_class == "match" and m.null_reasons == {}


def test_product_identities():
    m = compute_metrics(HAND_DF)
    assert m.recall_all == pytest.approx(m.recall_cond * m.coverage_pos)
    assert m.fpr_all == pytest.approx(m.fpr_cond * m.coverage_neg)
    assert m.correct_all == pytest.approx(m.correct_cond * m.coverage)


def test_matrix_rows_sum():
    assert confusion_2x3(HAND_DF, "match").tolist() == [[3, 1, 2], [1, 3, 0]]
    assert confusion_2x3(HAND_DF, "defect").tolist() == [[3, 1, 0], [1, 3, 2]]


def test_order_invariance():
    assert compute_metrics(HAND_DF.sample(frac=1, random_state=3)) == compute_metrics(HAND_DF)


def test_all_reject_run():                                   # Review Focus 1
    m = compute_metrics(ALL_REJECT)
    assert m.coverage == 0.0 and m.recall_all == 0.0 and m.fpr_all == 0.0 and m.correct_all == 0.0
    assert m.recall_cond is None and m.null_reasons["recall_cond"] == "undefined_no_accepted"
    assert m.fpr_cond is None and m.correct_cond is None


def test_empty_class_is_null():
    m = compute_metrics(_df([(1, "decided", 1), (1, "decided", 0)]))
    assert m.fpr_all is None and m.null_reasons["fpr_all"] == "undefined_empty_class"
    assert m.coverage_neg is None and m.fpr_cond is None and m.recall_all == 0.5


def test_not_applicable_excluded_from_Q():
    m = compute_metrics(pd.concat([HAND_DF, _df([(1, "not_applicable", None)] * 5)], ignore_index=True))
    assert m.n_not_applicable == 5 and m.n_pos == 6 and m.coverage == 0.8


def test_mixed_relations_raise():
    df = HAND_DF.copy()
    df.loc[0, "relation_id"] = "P3_SAME_PRODUCT"
    with pytest.raises(MixedRelationError):
        compute_metrics(df)


def test_bad_rows_rejected():
    for bad in (_df([(1, "error", None)]), _df([(1, "decided", None)]), _df([(2, "decided", 1)])):
        with pytest.raises(DataValidationError):
            compute_metrics(bad)


def test_defect_swap():
    m, m2 = compute_metrics(HAND_DF), compute_metrics(HAND_DF, "defect")
    assert m2.correct_all == m.correct_all and m2.recall_all == 3/4 and m2.fpr_all == 1/6
    assert m2.positive_class == "defect" and (m2.n_pos, m2.n_neg) == (m.n_neg, m.n_pos)
    assert (m2.tp, m2.tn, m2.fp, m2.fn) == (m.tn, m.tp, m.fn, m.fp) and (m2.rej_pos, m2.rej_neg) == (0, 2)
