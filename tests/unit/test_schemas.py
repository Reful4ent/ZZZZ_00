import copy

import pandas as pd
import pytest

from expcore.errors import DataValidationError
from expcore.schemas import (
    ALLOWED_SPLITS,
    IMAGES_COLUMNS,
    PAIRS_COLUMNS,
    PREDICTIONS_COLUMNS,
    validate_datasets_json,
    validate_images,
    validate_pairs,
    validate_predictions,
    validate_run_json,
)

pytestmark = pytest.mark.unit

H = "a" * 64


def make_images():
    rows = []
    for i, (group, split) in enumerate([("o1", "validation"), ("o1", "validation"), ("o2", "test"), ("o2", "test")]):
        rows.append(dict(image_id=f"kadid10k:{i:06d}", dataset_id="kadid10k", source_group_id=group,
                         class_id=None, category_id=None, split=split, relative_path=f"images/I{i}.png",
                         sha256=H, width=8, height=8, channels=3, defect_type=None))
    df = pd.DataFrame(rows)
    return df.astype({"width": "int32", "height": "int32", "channels": "int32"})


def make_pairs():
    return pd.DataFrame([
        dict(pair_id="p0", query_id="kadid10k:000000", reference_id="kadid10k:000001", relation_id="P1_SAME_SCENE",
             y_match=1, split="validation", source_group_id="o1", transform_id=None, severity=None, seed=None,
             gt_H_path=None, gt_mask_path=None, input_mask_path=None),
        dict(pair_id="p1", query_id="kadid10k:000002", reference_id="kadid10k:000003", relation_id="P1_SAME_SCENE",
             y_match=0, split="test", source_group_id="o2", transform_id=None, severity=None, seed=None,
             gt_H_path=None, gt_mask_path=None, input_mask_path=None),
    ]).astype({"y_match": "int8"})


def make_preds():
    return pd.DataFrame([
        dict(run_id="r", pair_id="p0", query_id=None, method_id="m", score_raw=0.0, score_direction="higher_is_match",
             y_pred=0, q_valid=True, reason_code=None, elapsed_ms=1.0, outcome="decided", timing_mode="per_item"),
        dict(run_id="r", pair_id="p1", query_id=None, method_id="m", score_raw=None, score_direction="lower_is_match",
             y_pred=None, q_valid=False, reason_code="REJECT_GENERIC", elapsed_ms=2.0, outcome="reject",
             timing_mode="per_item"),
    ])


def expect(inv, fn, *args, **kw):
    with pytest.raises(DataValidationError, match=inv):
        fn(*args, **kw)


def test_column_specs_cover_contract():
    assert {"image_id", "sha256", "defect_type"} <= set(IMAGES_COLUMNS)
    assert {"pair_id", "y_match", "input_mask_path"} <= set(PAIRS_COLUMNS)
    assert {"outcome", "timing_mode", "q_valid"} <= set(PREDICTIONS_COLUMNS)
    assert ALLOWED_SPLITS["mvtec_ad"] == {"dev", "validation", "synthetic_test", "real_test"}


def test_valid_tables_pass():
    images, pairs = make_images(), make_pairs()
    validate_images(images)
    validate_pairs(pairs, images)
    validate_predictions(make_preds(), pairs, require_complete=True)
    validate_predictions(make_preds().iloc[:1], pairs, require_complete=False)


def test_sha256_format():
    for bad in ("A" * 64, "a" * 63):
        images = make_images(); images.loc[0, "sha256"] = bad
        expect("INV-01", validate_images, images)


def test_inv01_duplicate_id_and_dims():
    images = make_images(); images.loc[1, "image_id"] = images.loc[0, "image_id"]
    expect("INV-01", validate_images, images)
    images = make_images(); images.loc[0, "width"] = 0
    expect("INV-01", validate_images, images)
    images = make_images(); images.loc[0, "relative_path"] = "images\\I0.png"
    expect("INV-01", validate_images, images)


def test_split_not_allowed_for_dataset():
    images = make_images(); images["split"] = "train"
    expect("INV-01", validate_images, images)


def test_missing_and_null_columns():
    expect("INV-01", validate_images, make_images().drop(columns=["sha256"]))
    images = make_images(); images.loc[0, "source_group_id"] = None
    expect("INV-01", validate_images, images)


def test_inv04_group_in_two_splits():
    images = make_images(); images.loc[1, "split"] = "test"
    expect("INV-04", validate_images, images)


def test_inv02_cross_split_pair():
    pairs = make_pairs(); pairs.loc[1, "reference_id"] = "kadid10k:000001"   # reference in validation
    expect("INV-02", validate_pairs, pairs, make_images())
    pairs = make_pairs(); pairs.loc[0, "query_id"] = "kadid10k:999999"
    expect("INV-02", validate_pairs, pairs, make_images())


def test_inv03_pairs():
    pairs = make_pairs(); pairs.loc[1, "pair_id"] = "p0"
    expect("INV-03", validate_pairs, pairs, make_images())
    pairs = make_pairs(); pairs.loc[0, "relation_id"] = "P9"
    expect("INV-03", validate_pairs, pairs, make_images())
    pairs = make_pairs(); pairs.loc[0, "y_match"] = 2
    expect("INV-03", validate_pairs, pairs, make_images())


def test_inv05_reject_row_shape():
    p = make_preds(); p.loc[1, "y_pred"] = 1
    expect("INV-05", validate_predictions, p, make_pairs(), require_complete=True)
    p = make_preds(); p.loc[1, "q_valid"] = True
    expect("INV-05", validate_predictions, p, make_pairs(), require_complete=True)
    p = make_preds(); p.loc[1, "reason_code"] = "NOT_APPLICABLE_NO_REGION"
    expect("INV-05", validate_predictions, p, make_pairs(), require_complete=True)
    p = make_preds(); p.loc[0, "y_pred"] = None
    expect("INV-05", validate_predictions, p, make_pairs(), require_complete=True)


def test_inv06_nan_score():
    p = make_preds(); p.loc[0, "score_raw"] = float("nan")
    expect("INV-06", validate_predictions, p, make_pairs(), require_complete=True)


def test_nullable_vs_reason():
    p = make_preds(); p.loc[0, "score_raw"] = None
    expect("INV-06", validate_predictions, p, make_pairs(), require_complete=True)
    p.loc[0, "reason_code"] = "SCORE_UNDEFINED"
    validate_predictions(p, make_pairs(), require_complete=True)
    p = make_preds(); p.loc[0, "elapsed_ms"] = None
    expect("INV-06", validate_predictions, p, make_pairs(), require_complete=True)


def test_inv07_missing_duplicate_extra_keys():   # Review Focus 2
    pairs = make_preds()
    preds_missing = pairs.iloc[:1]
    preds_dup = pd.concat([pairs, pairs.iloc[:1]], ignore_index=True)
    preds_extra = pairs.copy(); preds_extra.loc[1, "pair_id"] = "p_unknown"
    for p in (preds_missing, preds_dup, preds_extra):
        expect("INV-07", validate_predictions, p, make_pairs(), require_complete=True)
    for p in (preds_dup, preds_extra):
        expect("INV-07", validate_predictions, p, make_pairs(), require_complete=False)


def test_inv07_exactly_one_key():
    p = make_preds(); p.loc[0, "query_id"] = "kadid10k:000000"
    expect("INV-07", validate_predictions, p, make_pairs(), require_complete=True)


def test_inv07_per_run_and_method():
    other = make_preds(); other["method_id"] = "m2"
    validate_predictions(pd.concat([make_preds(), other], ignore_index=True), make_pairs(), require_complete=True)
    expect("INV-07", validate_predictions, pd.concat([make_preds(), other.iloc[:1]], ignore_index=True),
           make_pairs(), require_complete=True)


def test_inv08_component_inf():
    p = make_preds(); p["comp_inliers"] = [1.0, float("inf")]
    expect("INV-08", validate_predictions, p, make_pairs(), require_complete=True)


def test_inv09_no_technical_error_rows():
    p = make_preds(); p.loc[1, "outcome"] = "error"
    expect("INV-09", validate_predictions, p, make_pairs(), require_complete=True)


def dataset_record():
    return dict(dataset_id="kadid10k", source_url="https://example.org", archive_sha256=H, license="x",
                discovered_counts={"images": 4}, manifest_version="v1", download_date="2026-10-01T00:00:00Z",
                archive_name="kadid10k.zip", archive_verified=True, license_file={"path": "LICENSE", "sha256": H},
                extra_archives=[], completeness=None)


def test_datasets_json():
    validate_datasets_json([dataset_record()])
    hp = dict(dataset_record(), dataset_id="hpatches", completeness="subset", subset_name="mini")
    validate_datasets_json([dataset_record(), hp])
    for broken in (dict(dataset_record(), archive_sha256="ABC"),
                   dict(dataset_record(), download_date="2026-10-01"),
                   dict(dataset_record(), unknown=1),
                   dict(hp, subset_name=None),
                   dict(dataset_record(), completeness="full")):
        with pytest.raises(DataValidationError, match="datasets.json"):
            validate_datasets_json([broken])
    rec = dataset_record(); del rec["license"]
    with pytest.raises(DataValidationError, match="license"):
        validate_datasets_json([rec])
    with pytest.raises(DataValidationError, match="INV-08"):
        validate_datasets_json([dict(dataset_record(), discovered_counts={"x": float("nan")})])


def run_record():
    return dict(schema_version="1", run_id="photometric-pilot-aaaaaaaaaaaa-s11", config_hash=H, git_commit="abc123",
                dirty_diff_hash=None, dataset_manifest_hash=H, seed=11,
                environment={"path": "environment.json", "sha256": H, "python": "3.13", "os": "x", "cpu": "x", "gpu": None},
                status="running", started_at="2026-10-24T10:00:00Z", finished_at=None, profile="pilot",
                module="photometric", command="run", protocol_version="1", oracle_mode=False, post_test_study=False,
                library_primitives=[{"name": "sift", "library": "opencv", "version": "4", "check_id": "C-01"}],
                attempts=[], failure=None)


def test_run_json():
    validate_run_json(run_record())
    validate_run_json(dict(run_record(), status="completed", finished_at="2026-10-24T11:00:00Z"))
    failed = dict(run_record(), status="failed", finished_at="2026-10-24T11:00:00Z",
                  failure={"type": "TechnicalError", "message": "boom", "unit_id": None})
    validate_run_json(failed)
    validate_run_json(dict(run_record(), profile="smoke", run_id="photometric-smoke-aaaaaaaaaaaa-s11", git_commit=None))
    for broken in (dict(run_record(), git_commit=None),
                   dict(run_record(), status="completed"),
                   dict(failed, failure=None),
                   dict(run_record(), status="done"),
                   dict(run_record(), config_hash="x"),
                   dict(run_record(), seed="11"),
                   dict(run_record(), extra=1),
                   dict(run_record(), seed=23)):   # run_id no longer matches D-08
        with pytest.raises(DataValidationError, match="run.json"):
            validate_run_json(broken)
    rec = copy.deepcopy(run_record()); del rec["environment"]["sha256"]
    with pytest.raises(DataValidationError, match="environment"):
        validate_run_json(rec)


def test_inv07_empty_predictions_incomplete():   # Review Focus 2: no rows at all is not "complete"
    empty = make_preds().iloc[:0]
    expect("INV-07", validate_predictions, empty, make_pairs(), require_complete=True)
    validate_predictions(empty, make_pairs(), require_complete=False)


def test_datasets_json_unverified_archive_allows_nulls():   # user decision 1a
    unverified = dict(dataset_record(), archive_verified=False, archive_sha256=None, download_date=None,
                      license_file=None)
    validate_datasets_json([unverified])
    for field in ("archive_sha256", "download_date", "license_file"):
        with pytest.raises(DataValidationError, match=field):
            validate_datasets_json([dict(dataset_record(), **{field: None})])   # verified: still required
