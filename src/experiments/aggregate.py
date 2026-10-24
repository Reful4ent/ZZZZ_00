"""experiments aggregate (spec §6.3, §7.1, §7.7): §6 metrics per (method, relation, split, seed) of every completed run."""

import dataclasses
import importlib
from pathlib import Path

import pandas as pd

from expcore.cli_common import PROGRAM_OF
from expcore.config import load_config
from expcore.errors import ProtocolViolationError
from expcore.hashing import canonical_json
from expcore.io import load_json, write_parquet
from expcore.loaders import accessible_splits, load_pair_evaluation
from expcore.manifests import load_manifest, manifest_version
from expcore.metrics import MetricsRow, compute_metrics
from expcore.resume import SUCCESS, read_predictions, verify_shards
from expcore.schemas import validate_predictions
from experiments.plan import load_plan

TRANSFORM = {"higher_is_match": "identity", "lower_is_match": "negate"}
QUERY_METRICS_REASON = "query_metrics_defined_by_tz_03"
_COUNTS = [f.name for f in dataclasses.fields(MetricsRow) if f.type in (int, "int")]
_RATIOS = [f.name for f in dataclasses.fields(MetricsRow) if f.name not in _COUNTS
           and f.name not in ("positive_class", "null_reasons")]
METRICS_SCHEMA = {
    "run_id": "str", "module": "str", "method_id": "str", "relation_id": "str", "split": "str", "seed": "int64",
    "profile": "str", "section": "str", "oracle_mode": "bool", "post_test_study": "bool", "positive_class": "str", "score_transform": "str",
    "n_pairs": "int64", "n_source_groups": "int64", "n_seeds": "int64",
    **dict.fromkeys(_COUNTS, "int64"), **dict.fromkeys(_RATIOS, "float64"), "null_reasons": "str",
}


def run_dir_of(cfg, profile: str, module: str, run_id: str) -> Path:
    return cfg.paths.out_root / profile / module / run_id


def _is_completed(run_dir: Path) -> bool:
    run_json = run_dir / "run.json"
    return (run_dir / SUCCESS).exists() and run_json.exists() and load_json(run_json)["status"] == "completed"


def run_statuses(plan_dir: Path) -> dict[str, str]:
    """run_id → run.json status ("not started" without a run directory), for every planned run."""
    plan_dir = Path(plan_dir)
    profile = load_json(plan_dir / "plan.json")["profile"]
    out = {}
    for r in load_plan(plan_dir):
        importlib.import_module(f"{r.module}.cli")
        run_dir = run_dir_of(load_config(plan_dir / r.config_path, module=r.module), profile, r.module, r.run_id)
        out[r.run_id] = "completed" if _is_completed(run_dir) else (
            load_json(run_dir / "run.json")["status"] if (run_dir / "run.json").exists() else "not started")
    return out


def aggregate(plan_dir: Path, out_path: Path) -> pd.DataFrame:
    """Every completed run of the plan; incomplete predictions → DataValidationError (exit 3), never rejects."""
    plan_dir = Path(plan_dir)
    profile = load_json(plan_dir / "plan.json")["profile"]
    rows = []
    for r in load_plan(plan_dir):
        importlib.import_module(f"{r.module}.cli")   # registers the module's params schema
        cfg = load_config(plan_dir / r.config_path, module=r.module)
        run_dir = run_dir_of(cfg, profile, r.module, r.run_id)
        if not _is_completed(run_dir):
            continue
        run = load_json(run_dir / "run.json")
        if run["profile"] != profile or run["run_id"] != r.run_id:
            raise ProtocolViolationError(f"{run_dir}: run.json does not belong to this {profile} plan")
        verify_shards(run_dir)
        manifest = load_manifest(cfg.paths.manifest_dir, manifest_version(cfg))
        if run["dataset_manifest_hash"] != manifest.manifest_hash:
            raise ProtocolViolationError(f"{run_dir}: run used another manifest than {manifest.version}")
        pairs = manifest.pairs[PROGRAM_OF[r.module]]
        splits = accessible_splits(profile, fixture=all(d.fixture for d in cfg.datasets.values()),
                                   splits=set(pairs["split"]))
        pairs = pairs[pairs["split"].isin(splits)]
        pred = read_predictions(run_dir)
        base = {"run_id": r.run_id, "module": r.module, "seed": r.seed, "profile": profile,
                "section": "oracle" if run["oracle_mode"] else "main", "oracle_mode": run["oracle_mode"],
                "post_test_study": run["post_test_study"]}
        if r.module == "metric":   # query-level predictions: completeness only, metrics come with TZ 03
            validate_predictions(pred, pd.DataFrame({"query_id": sorted(set(pairs["query_id"]))}),
                                 require_complete=True)
            for method_id, g in pred.groupby("method_id"):
                rows.append({**base, "method_id": method_id, "relation_id": "P3_SAME_PRODUCT", "split": "all",
                             "positive_class": "match", "score_transform": _transform(g), "n_pairs": len(g),
                             "n_source_groups": int(pairs["source_group_id"].nunique()),
                             **dict.fromkeys(_COUNTS, None), **dict.fromkeys(_RATIOS, None),
                             "null_reasons": canonical_json(dict.fromkeys(_RATIOS, QUERY_METRICS_REASON))})
            continue
        validate_predictions(pred, pairs, require_complete=True)
        evaluation = pd.concat([load_pair_evaluation(pairs, s) for s in splits], ignore_index=True)
        joined = pred.drop(columns=["query_id"]).merge(evaluation, on="pair_id", how="inner")
        for (method_id, relation_id, split), g in joined.groupby(["method_id", "relation_id", "split"]):
            m = dataclasses.asdict(compute_metrics(g[["y_match", "outcome", "y_pred", "relation_id"]]))
            reasons = m.pop("null_reasons")
            rows.append({**base, "method_id": method_id, "relation_id": relation_id, "split": split,
                         "score_transform": _transform(g), "n_pairs": len(g),
                         "n_source_groups": int(g["source_group_id"].nunique()), **m,
                         "null_reasons": canonical_json(reasons)})
    df = pd.DataFrame(rows, columns=[c for c in METRICS_SCHEMA if c != "n_seeds"])
    keys = ["section", "module", "method_id", "relation_id", "split"]
    df["n_seeds"] = df.groupby(keys)["seed"].transform("nunique").astype("int64") if len(df) else []
    df = df[list(METRICS_SCHEMA)].sort_values(keys + ["seed"], kind="mergesort").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, out_path, schema=METRICS_SCHEMA, nullable=frozenset(_COUNTS + _RATIOS))
    return df


def _transform(g: pd.DataFrame) -> str:
    kinds = sorted({TRANSFORM[d] for d in g["score_direction"]})
    return kinds[0] if len(kinds) == 1 else "mixed"
