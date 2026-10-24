"""Data contract (spec §5.1–5.5) and invariants INV-01…INV-09; every violation names its invariant."""

import json
import re
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from expcore.errors import DataValidationError
from expcore.registry import RELATIONS, Outcome, is_registered

IMAGES_COLUMNS = {
    "image_id": "str", "dataset_id": "str", "source_group_id": "str", "class_id": "str",
    "category_id": "str", "split": "str", "relative_path": "str", "sha256": "str",
    "width": "int32", "height": "int32", "channels": "int32", "defect_type": "str",
}
IMAGES_NULLABLE = frozenset({"class_id", "category_id", "defect_type"})

PAIRS_COLUMNS = {
    "pair_id": "str", "query_id": "str", "reference_id": "str", "relation_id": "str",
    "y_match": "int8", "split": "str", "source_group_id": "str", "transform_id": "str",
    "severity": "float64", "seed": "int64", "gt_H_path": "str", "gt_mask_path": "str",
    "input_mask_path": "str",
}
PAIRS_NULLABLE = frozenset({"transform_id", "severity", "seed", "gt_H_path", "gt_mask_path", "input_mask_path"})

PREDICTIONS_COLUMNS = {
    "run_id": "str", "pair_id": "str", "query_id": "str", "method_id": "str",
    "score_raw": "float64", "score_direction": "str", "y_pred": "int8", "q_valid": "bool",
    "reason_code": "str", "elapsed_ms": "float64", "outcome": "str", "timing_mode": "str",
}
PREDICTIONS_NULLABLE = frozenset({"pair_id", "query_id", "score_raw", "y_pred", "reason_code", "elapsed_ms"})
COMPONENT_PREFIX = "comp_"  # comp_* columns: nullable float64 decision components

ALLOWED_SPLITS: dict[str, frozenset[str]] = {
    "kadid10k": frozenset({"validation", "test"}),
    "hpatches": frozenset({"validation", "test"}),
    "mvtec_ad": frozenset({"dev", "validation", "synthetic_test", "real_test"}),
    "sop": frozenset({"train", "validation", "test"}),
}
EVAL_ONLY_PAIR_COLUMNS = ("y_match", "gt_H_path", "gt_mask_path")
EVAL_ONLY_IMAGE_COLUMNS = ("defect_type",)

SCORE_DIRECTIONS = ("higher_is_match", "lower_is_match")
TIMING_MODES = ("per_item", "amortized_batch")
PROFILES = ("smoke", "pilot", "confirmatory")
RUN_STATUSES = ("running", "completed", "failed")
COMPLETENESS = ("full", "subset", "unverified")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")


def _fail(inv: str, msg: str) -> None:
    raise DataValidationError(f"{inv}: {msg}")


def _dtype_ok(s: pd.Series, dtype: str) -> bool:
    values = s.dropna()
    if values.empty:
        return True
    if dtype == "str":
        return bool(values.map(lambda v: isinstance(v, str)).all())
    if dtype == "bool":
        return pd.api.types.is_bool_dtype(s)
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return False
    if dtype.startswith("int"):
        return bool((values.astype(float) % 1 == 0).all())
    return True


def _check_table(df: pd.DataFrame, columns: Mapping[str, str], nullable: frozenset, inv: str, table: str,
                 extra_ok: Callable[[str], bool] = lambda c: False) -> None:
    missing = [c for c in columns if c not in df.columns]
    unknown = [c for c in df.columns if c not in columns and not extra_ok(c)]
    if missing or unknown:
        _fail(inv, f"{table}: missing columns {missing}, unknown columns {unknown}")
    for col, dtype in columns.items():
        if col not in nullable and df[col].isna().any():
            _fail(inv, f"{table}.{col}: null values in a non-nullable column")
        if not _dtype_ok(df[col], dtype):
            _fail(inv, f"{table}.{col}: values are not of type {dtype}")


def _finite_or_null(s: pd.Series) -> pd.Series:
    """True where the value is null or finite."""
    v = s.to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(np.isnan(v) | np.isfinite(v), index=s.index)


def validate_images(images: pd.DataFrame) -> None:
    _check_table(images, IMAGES_COLUMNS, IMAGES_NULLABLE, "INV-01", "images")
    unknown = set(images["dataset_id"]) - set(ALLOWED_SPLITS)
    if unknown:
        _fail("INV-01", f"images.dataset_id: unknown {sorted(unknown)}")
    dup = images["image_id"][images["image_id"].duplicated()]
    if not dup.empty:
        _fail("INV-01", f"images.image_id: duplicates {dup.head().tolist()}")
    bad = images["sha256"][~images["sha256"].str.fullmatch(_SHA256.pattern)]
    if not bad.empty:
        _fail("INV-01", f"images.sha256: not 64 lowercase hex: {bad.head().tolist()}")
    for col in ("width", "height", "channels"):
        if (images[col] <= 0).any():
            _fail("INV-01", f"images.{col}: values must be > 0")
    path = images["relative_path"]
    bad = path[path.str.contains("\\", regex=False) | path.str.match(r"/|[A-Za-z]:")]
    if not bad.empty:
        _fail("INV-01", f"images.relative_path: must be relative with '/' separators: {bad.head().tolist()}")
    for dataset_id, g in images.groupby("dataset_id"):
        bad = g["split"][~g["split"].isin(ALLOWED_SPLITS[dataset_id])]
        if not bad.empty:
            _fail("INV-01", f"images.split: {sorted(set(bad))} not allowed for {dataset_id}")
    n_splits = images.groupby(["dataset_id", "source_group_id"])["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if not leaked.empty:
        _fail("INV-04", f"source groups in several splits: {leaked.index[:5].tolist()}")


def validate_pairs(pairs: pd.DataFrame, images: pd.DataFrame) -> None:
    _check_table(pairs, PAIRS_COLUMNS, PAIRS_NULLABLE, "INV-03", "pairs")
    dup = pairs["pair_id"][pairs["pair_id"].duplicated()]
    if not dup.empty:
        _fail("INV-03", f"pairs.pair_id: duplicates {dup.head().tolist()}")
    bad = pairs["relation_id"][~pairs["relation_id"].isin(RELATIONS)]
    if not bad.empty:
        _fail("INV-03", f"pairs.relation_id: unregistered {sorted(set(bad))}")
    if not pairs["y_match"].isin([0, 1]).all():
        _fail("INV-03", "pairs.y_match: values must be 0 or 1")
    split_of = images.set_index("image_id")["split"]
    q_split, r_split = pairs["query_id"].map(split_of), pairs["reference_id"].map(split_of)
    absent = pairs["pair_id"][q_split.isna() | r_split.isna()]
    if not absent.empty:
        _fail("INV-02", f"pairs reference unknown images: {absent.head().tolist()}")
    cross = pairs["pair_id"][(q_split != pairs["split"]) | (r_split != pairs["split"])]
    if not cross.empty:
        _fail("INV-02", f"cross-split pairs: {cross.head().tolist()}")


def validate_predictions(pred: pd.DataFrame, pairs: pd.DataFrame, *, require_complete: bool) -> None:
    is_comp = lambda c: isinstance(c, str) and c.startswith(COMPONENT_PREFIX)  # noqa: E731
    comp_cols = [c for c in pred.columns if is_comp(c)]
    _check_table(pred, PREDICTIONS_COLUMNS | {c: "float64" for c in comp_cols},
                 PREDICTIONS_NULLABLE | frozenset(comp_cols), "INV-05", "predictions", is_comp)

    bad = set(pred["outcome"]) - {o.value for o in Outcome}
    if bad:
        _fail("INV-09", f"predictions.outcome: {sorted(bad)} — technical errors and unknown outcomes "
                        "do not belong in predictions")
    for col, allowed in (("score_direction", SCORE_DIRECTIONS), ("timing_mode", TIMING_MODES)):
        bad = set(pred[col]) - set(allowed)
        if bad:
            _fail("INV-05", f"predictions.{col}: {sorted(bad)} not in {allowed}")

    decided = pred["outcome"] == Outcome.DECIDED.value
    y_ok = pred["y_pred"].isin([0, 1])
    shape_bad = (decided & (~y_ok | ~pred["q_valid"])) | (~decided & (pred["y_pred"].notna() | pred["q_valid"]))
    if shape_bad.any():
        _fail("INV-05", f"y_pred/q_valid inconsistent with outcome in rows {pred.index[shape_bad][:5].tolist()}")
    if pred["reason_code"][~decided].isna().any():
        _fail("INV-05", "reject/not_applicable rows require a reason_code")
    coded = pred[["reason_code", "outcome"]].dropna().drop_duplicates()
    for code, outcome in coded.itertuples(index=False, name=None):
        if not is_registered(code, Outcome(outcome)):
            _fail("INV-05", f"reason_code {code!r} is not registered for outcome {outcome!r}")

    no_reason = pred["reason_code"].isna()
    if not _finite_or_null(pred["score_raw"]).all():
        _fail("INV-06", "predictions.score_raw: NaN/Inf")
    if (pred["score_raw"].isna() & no_reason).any():
        _fail("INV-06", "predictions.score_raw: null requires a reason_code")
    if (pred["elapsed_ms"].isna() & no_reason).any():
        _fail("INV-06", "predictions.elapsed_ms: null requires a reason_code")
    for col in ["elapsed_ms", *comp_cols]:
        if not _finite_or_null(pred[col]).all():
            _fail("INV-08", f"predictions.{col}: Inf")

    if pred.empty:
        if require_complete and not pairs.empty:
            _fail("INV-07", f"no predictions for an evaluated set of {len(pairs)} rows")
        return
    has_pair, has_query = pred["pair_id"].notna(), pred["query_id"].notna()
    if (has_pair == has_query).any():
        _fail("INV-07", "exactly one of pair_id / query_id must be set in every row")
    for (run_id, method_id), g in pred.groupby(["run_id", "method_id"]):
        where = f"run_id={run_id!r}, method_id={method_id!r}"
        if g["pair_id"].notna().all():
            key = "pair_id"
        elif g["query_id"].notna().all():
            key = "query_id"
        else:
            _fail("INV-07", f"{where}: pair and query rows mixed")
        keys = g[key]
        expected = set(pairs[key].dropna())
        dup = keys[keys.duplicated()]
        if not dup.empty:
            _fail("INV-07", f"{where}: duplicate keys {dup.head().tolist()}")
        extra = set(keys) - expected
        if extra:
            _fail("INV-07", f"{where}: keys absent from the evaluated set {sorted(extra)[:5]}")
        if require_complete:
            missing = expected - set(keys)
            if missing:
                _fail("INV-07", f"{where}: {len(missing)} keys without a prediction, e.g. {sorted(missing)[:5]}")


# --- JSON records -----------------------------------------------------------------------------

_Check = Callable[[Any], bool]
_SHA_SCHEMA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_TS_SCHEMA = {"type": "string", "pattern": "^" + _TS.pattern + "$"}


def _check(fn: Callable[[Any], bool], schema: dict) -> _Check:
    """A value check that also carries its JSON Schema fragment (for scripts/gen_schemas.py)."""
    fn.schema = schema
    return fn


_is_str = _check(lambda v: isinstance(v, str), {"type": "string"})
_is_int = _check(lambda v: isinstance(v, int) and not isinstance(v, bool), {"type": "integer"})
_is_bool = _check(lambda v: isinstance(v, bool), {"type": "boolean"})
_is_obj = _check(lambda v: isinstance(v, dict), {"type": "object"})
_is_list = _check(lambda v: isinstance(v, list), {"type": "array"})
_is_sha = _check(lambda v: isinstance(v, str) and _SHA256.fullmatch(v) is not None, _SHA_SCHEMA)
_is_ts = _check(lambda v: isinstance(v, str) and _TS.fullmatch(v) is not None, _TS_SCHEMA)


def _nullable(check: _Check) -> _Check:
    return _check(lambda v: v is None or check(v), {"anyOf": [check.schema, {"type": "null"}]})


def _one_of(*allowed: str) -> _Check:
    return _check(lambda v: v in allowed, {"enum": list(allowed)})


def _record(**fields: _Check) -> _Check:
    """An object with exactly these keys, each passing its check."""
    return _check(lambda v: isinstance(v, dict) and set(v) == set(fields) and all(c(v[k]) for k, c in fields.items()),
                  {"type": "object", "required": sorted(fields), "additionalProperties": False,
                   "properties": {k: c.schema for k, c in fields.items()}})


def _keys(*names: str) -> _Check:
    return _check(lambda v: isinstance(v, dict) and set(v) == set(names),
                  {"type": "object", "required": sorted(names), "additionalProperties": False,
                   "properties": dict.fromkeys(names, {})})


def _list_of(check: _Check) -> _Check:
    return _check(lambda v: isinstance(v, list) and all(check(x) for x in v), {"type": "array", "items": check.schema})


def _check_json_record(rec: Any, fields: Mapping[str, tuple[_Check, bool]], where: str,
                       optional: frozenset = frozenset()) -> None:
    """fields: name -> (check, nullable). Every field required; unknown keys rejected."""
    if not isinstance(rec, dict):
        _fail(where, "record must be an object")
    missing = [k for k in fields if k not in rec and k not in optional]
    unknown = [k for k in rec if k not in fields]
    if missing or unknown:
        _fail(where, f"missing keys {missing}, unknown keys {unknown}")
    for name, (check, nullable) in fields.items():
        if name not in rec:
            continue
        value = rec[name]
        if value is None:
            if not nullable:
                _fail(where, f"{name}: must not be null")
        elif not check(value):
            _fail(where, f"{name}: invalid value {value!r}")


def record_json_schema(fields: Mapping[str, tuple[_Check, bool]], optional: frozenset = frozenset()) -> dict:
    """JSON Schema of a record validated by _check_json_record (same fields, nullability, strictness)."""
    props = {k: ({"anyOf": [c.schema, {"type": "null"}]} if nullable else c.schema)
             for k, (c, nullable) in fields.items()}
    return {"type": "object", "required": [k for k in fields if k not in optional], "additionalProperties": False,
            "properties": props}


def _no_nan(obj: Any, where: str) -> None:
    try:
        json.dumps(obj, allow_nan=False)
    except ValueError as e:
        raise DataValidationError(f"INV-08: {where}: NaN/Inf in JSON") from e


DATASET_FIELDS = {
    "dataset_id": (_one_of(*ALLOWED_SPLITS), False),
    "source_url": (_is_str, False),
    "archive_sha256": (_is_sha, True),   # null only when archive_verified is false
    "license": (_is_str, False),
    "discovered_counts": (_is_obj, False),
    "manifest_version": (_is_str, False),
    "download_date": (_is_ts, True),
    "archive_name": (_is_str, False),
    "archive_verified": (_is_bool, False),
    "license_file": (_record(path=_nullable(_is_str), sha256=_nullable(_is_sha)), True),
    "extra_archives": (_list_of(_record(name=_is_str, sha256=_is_sha)), False),
    "completeness": (_one_of(*COMPLETENESS), True),
    "subset_name": (_is_str, False),
}
DATASET_OPTIONAL = frozenset({"subset_name"})


# Provenance that may be unknown for unverified archives (smoke fixtures, missing downloads).
_VERIFIED_REQUIRED = ("archive_sha256", "download_date", "license_file")


def validate_datasets_json(obj: list[dict]) -> None:
    _no_nan(obj, "datasets.json")
    if not isinstance(obj, list):
        _fail("datasets.json", "must be a JSON array")
    seen = set()
    for i, rec in enumerate(obj):
        where = f"datasets.json[{i}]"
        _check_json_record(rec, DATASET_FIELDS, where, optional=DATASET_OPTIONAL)
        if rec["dataset_id"] in seen:
            _fail(where, f"duplicate dataset_id {rec['dataset_id']!r}")
        seen.add(rec["dataset_id"])
        if rec["archive_verified"]:
            unset = [k for k in _VERIFIED_REQUIRED if rec[k] is None]
            if unset:
                _fail(where, f"{unset}: required when archive_verified is true")
        if (rec["dataset_id"] == "hpatches") != (rec["completeness"] is not None):
            _fail(where, "completeness is required for hpatches and must be null for other datasets")
        if (rec["completeness"] == "subset") != ("subset_name" in rec):
            _fail(where, "subset_name is required exactly when completeness is 'subset'")


RUN_FIELDS = {
    "schema_version": (_is_str, False),
    "run_id": (_is_str, False),
    "config_hash": (_is_sha, False),
    "git_commit": (_is_str, True),
    "dirty_diff_hash": (_is_sha, True),
    "dataset_manifest_hash": (_is_sha, False),
    "seed": (_is_int, False),
    "environment": (_record(path=_is_str, sha256=_is_sha, python=_nullable(_is_str), os=_nullable(_is_str),
                            cpu=_nullable(_is_str), gpu=_check(lambda v: True, {})), False),
    "status": (_one_of(*RUN_STATUSES), False),
    "started_at": (_is_ts, False),
    "finished_at": (_is_ts, True),
    "profile": (_one_of(*PROFILES), False),
    "module": (_is_str, False),
    "command": (_is_str, False),
    "protocol_version": (_is_str, False),
    "oracle_mode": (_is_bool, False),
    "post_test_study": (_is_bool, False),
    "library_primitives": (_list_of(_record(name=_is_str, library=_is_str, version=_nullable(_is_str),
                                            check_id=_is_str)), False),
    "attempts": (_is_list, False),
    "failure": (_record(type=_is_str, message=_is_str, unit_id=_nullable(_is_int)), True),
    "paths_resolved": (_is_obj, False),  # §8.2: actual paths, outside config_hash
}
RUN_OPTIONAL = frozenset({"paths_resolved"})


def validate_run_json(obj: dict) -> None:
    _no_nan(obj, "run.json")
    _check_json_record(obj, RUN_FIELDS, "run.json", optional=RUN_OPTIONAL)
    expected_id = f"{obj['module']}-{obj['profile']}-{obj['config_hash'][:12]}-s{obj['seed']}"
    if obj["run_id"] != expected_id:
        _fail("run.json", f"run_id {obj['run_id']!r} != {expected_id!r} (D-08)")
    if obj["git_commit"] is None and obj["profile"] != "smoke":
        _fail("run.json", "git_commit may be null only in the smoke profile")
    if (obj["status"] == "running") != (obj["finished_at"] is None):
        _fail("run.json", "finished_at must be null exactly while status is running")
    if (obj["status"] == "failed") != (obj["failure"] is not None):
        _fail("run.json", "failure must be set exactly when status is failed")
