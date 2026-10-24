"""Runner with atomic shards and resume without rewriting (spec §3.4, §7.6, ПОСТ-01).

Unit of work = chunk of the key-sorted items (pair_id / query_id); unit_id is the chunk number.
A unit is done when its shard exists and its SHA-256 equals the one in progress.json.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from expcore.config import Config
from expcore.decisions import Decision
from expcore.errors import DataValidationError, ProtocolError, ResultTamperedError, TechnicalError
from expcore.hashing import canonical_json, sha256_file, table_content_hash
from expcore.io import atomic_write_bytes, dump_json, load_json, read_parquet, write_parquet
from expcore.loaders import ImageStore, PairInput, QueryInput
from expcore.runlog import JsonlLogger, RunRecord, now_utc
from expcore.schemas import COMPONENT_PREFIX, PREDICTIONS_COLUMNS, PREDICTIONS_NULLABLE, validate_predictions
from expcore.seeding import capture_rng_state, stream
from expcore.timing import timed

SUCCESS = "_SUCCESS"
AUG_COLUMNS = ("unit_id", "sample_id", "aug_name", "params_json")


class AugmentationLogger:
    """Rows for augmentations.parquet; a unit's rows are kept only when the unit completes."""

    def __init__(self, run_dir: Path) -> None:
        self._path = Path(run_dir) / "augmentations.parquet"
        self._pending: list[dict] = []

    def log(self, unit_id: int, sample_id: str, aug_name: str, params: Any) -> None:
        self._pending.append({"unit_id": unit_id, "sample_id": str(sample_id), "aug_name": aug_name,
                              "params_json": canonical_json(params)})

    def commit(self) -> None:
        old = read_parquet(self._path) if self._path.exists() else pd.DataFrame(columns=list(AUG_COLUMNS))
        new = pd.DataFrame(self._pending, columns=list(AUG_COLUMNS))
        table = pd.concat([old, new], ignore_index=True) if len(old) else new
        write_parquet(table, self._path, schema={"unit_id": "int64", "sample_id": "str", "aug_name": "str",
                                                 "params_json": "str"})
        self._pending = []

    def discard(self) -> None:
        self._pending = []


@dataclass
class RunContext:
    run_id: str
    seed: int
    profile: str
    run_dir: Path
    cfg: Config
    record: RunRecord
    # Set by the Runner for the unit being computed: algorithms draw randomness from ctx.rng only.
    unit_id: int | None = None
    rng: np.random.Generator | None = None
    augmentations: AugmentationLogger | None = None


def read_predictions(run_dir: Path) -> pd.DataFrame:
    """All completed shards; temporary files (part-*.parquet.tmp-<pid>) are never read."""
    parts = sorted((Path(run_dir) / "predictions").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=list(PREDICTIONS_COLUMNS))
    return pd.concat([read_parquet(p) for p in parts], ignore_index=True)


def predictions_content_hash(df: pd.DataFrame) -> str:
    """Content hash for T-REP-02; elapsed_ms is a measurement, not a result, and is left out."""
    comps = sorted(c for c in df.columns if str(c).startswith(COMPONENT_PREFIX))
    columns = [c for c in PREDICTIONS_COLUMNS if c != "elapsed_ms"] + comps
    return table_content_hash(df, columns, ["run_id", "method_id", "pair_id", "query_id"])


def _resolved_config(cfg: Config) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


class Runner:
    def __init__(self, ctx: RunContext, predictor: Any, images: ImageStore) -> None:
        self.ctx = ctx
        self.predictor = predictor
        self.images = images
        self.pred_dir = ctx.run_dir / "predictions"
        self.log = JsonlLogger(ctx.run_dir / "run.log.jsonl", ctx.run_id)
        self.errors = JsonlLogger(ctx.run_dir / "errors.jsonl", ctx.run_id)

    # --- bookkeeping -------------------------------------------------------------------------

    def _progress(self) -> dict:
        path = self.ctx.run_dir / "progress.json"
        return load_json(path) if path.exists() else {"units": {}}

    def _verify_done(self, progress: dict) -> None:
        for unit, entry in progress["units"].items():
            shard = self.pred_dir / f"part-{int(unit):06d}.parquet"
            if not shard.is_file() or sha256_file(shard) != entry["sha256"]:
                raise ResultTamperedError(f"{shard}: completed shard is missing or its SHA-256 changed; "
                                          "completed results are never recomputed — remove the run directory")

    def _write_once(self, name: str, content: Any) -> None:
        """Write a run-level artifact on the first attempt; a resume must see the same content."""
        path = self.ctx.run_dir / name
        if path.exists():
            if load_json(path) != json.loads(json.dumps(content)):
                raise ResultTamperedError(f"{path}: differs from this attempt; the run's inputs changed")
            return
        dump_json(content, path)

    # --- running -----------------------------------------------------------------------------

    def run(self, items: Sequence[PairInput | QueryInput], *, method_id: str) -> Path:
        ctx = self.ctx
        is_query = bool(items) and isinstance(items[0], QueryInput)
        key = "query_id" if is_query else "pair_id"
        items = sorted(items, key=lambda it: getattr(it, key))
        keys = [getattr(it, key) for it in items]
        if len(set(keys)) != len(keys):
            raise DataValidationError(f"run items have duplicate {key} values")

        if (ctx.run_dir / SUCCESS).exists():   # completed: verify, never rewrite
            self._verify_done(self._progress())
            return self.pred_dir
        ctx.record.start()
        try:
            return self._run(items, keys, key, is_query, method_id)
        except Exception as e:
            failure = {"type": type(e).__name__ if isinstance(e, ProtocolError) else "TechnicalError",
                       "message": str(e), "unit_id": ctx.unit_id}
            self.errors.log("ERROR", "run_failed", **failure)
            self.log.log("ERROR", "run_failed", **failure)
            ctx.record.finish("failed", failure)
            if isinstance(e, ProtocolError):
                raise
            raise TechnicalError(f"unit {ctx.unit_id}: {type(e).__name__}: {e}") from e

    def _run(self, items: list, keys: list[str], key: str, is_query: bool, method_id: str) -> Path:
        ctx = self.ctx
        chunk = ctx.cfg.runner.chunk_size
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        for leftover in list(self.pred_dir.glob("*.tmp-*")) + list(ctx.run_dir.glob("*.tmp-*")):
            leftover.unlink()   # Review Focus 3: a temp file from a killed process is never a result
        self._write_once("file_order.json", {"key": key, "method_id": method_id, "chunk_size": chunk, "keys": keys})
        config_path = ctx.run_dir / "config.resolved.yaml"
        if not config_path.exists():
            atomic_write_bytes(config_path, yaml.safe_dump(_resolved_config(ctx.cfg), allow_unicode=True,
                                                           sort_keys=True).encode("utf-8"))
        rng_path = ctx.run_dir / "rng_state.json"
        rng_state = load_json(rng_path) if rng_path.exists() else {"start": capture_rng_state(), "units": {}}
        dump_json(rng_state, rng_path)
        progress = self._progress()
        self._verify_done(progress)
        ctx.augmentations = AugmentationLogger(ctx.run_dir)
        timers = {"started_at": now_utc(), "compute_ms": 0.0, "write_ms": 0.0, "units_computed": 0}

        for unit_id in range(0, (len(items) + chunk - 1) // chunk):
            if str(unit_id) in progress["units"]:
                continue
            shard = self.pred_dir / f"part-{unit_id:06d}.parquet"
            shard.unlink(missing_ok=True)   # written but not recorded in progress.json: not completed
            ctx.unit_id, ctx.rng = unit_id, stream(ctx.seed, "sampler", unit_id)
            unit_items = items[unit_id * chunk:(unit_id + 1) * chunk]
            rows, ms = self._predict_unit(unit_items, key, is_query, method_id)
            timers["compute_ms"] += ms
            df = self._to_frame(rows)
            comps = [c for c in df.columns if c.startswith(COMPONENT_PREFIX)]
            _, write_ms = timed(lambda: write_parquet(
                df, shard, schema=PREDICTIONS_COLUMNS | dict.fromkeys(comps, "float64"),
                nullable=PREDICTIONS_NULLABLE | frozenset(comps)))
            timers["write_ms"] += write_ms
            ctx.augmentations.commit()
            progress["units"][str(unit_id)] = {"unit_id": unit_id, "n_rows": len(df), "sha256": sha256_file(shard)}
            dump_json(progress, ctx.run_dir / "progress.json")
            rng_state["units"][str(unit_id)] = capture_rng_state()
            dump_json(rng_state, rng_path)
            timers["units_computed"] += 1
            self.log.log("INFO", "unit_done", unit_id=unit_id, n_rows=len(df))
        ctx.unit_id = None

        expected = pd.DataFrame({key: keys})
        validate_predictions(read_predictions(ctx.run_dir), expected, require_complete=True)
        if not (ctx.run_dir / "augmentations.parquet").exists():
            ctx.augmentations.commit()
        timers_path = ctx.run_dir / "timers.json"
        all_timers = load_json(timers_path) if timers_path.exists() else {"attempts": []}
        timers["total_ms"] = timers["compute_ms"] + timers["write_ms"]
        all_timers["attempts"].append(timers)
        dump_json(all_timers, timers_path)
        atomic_write_bytes(ctx.run_dir / SUCCESS, b"")
        ctx.record.finish("completed")
        self.log.log("INFO", "run_completed", n_rows=len(keys))
        return self.pred_dir

    def _predict_unit(self, unit_items: list, key: str, is_query: bool, method_id: str) -> tuple[list[dict], float]:
        ctx = self.ctx
        if is_query:
            decisions, ms = timed(self.predictor.predict, unit_items, self.images, ctx)
            decisions = list(decisions)
            if len(decisions) != len(unit_items):
                raise TechnicalError(f"query predictor returned {len(decisions)} decisions for {len(unit_items)} queries")
            per = ms / len(unit_items)
            timed_decisions = [(d, per, "amortized_batch") for d in decisions]
        else:
            timed_decisions = []
            for item in unit_items:
                d, item_ms = timed(self.predictor.predict, item, self.images, ctx)
                timed_decisions.append((d, item_ms, "per_item"))
        rows = []
        for item, (d, item_ms, mode) in zip(unit_items, timed_decisions):
            if not isinstance(d, Decision):
                raise TechnicalError(f"predictor returned {type(d).__name__}, expected Decision")
            rows.append({
                "run_id": ctx.run_id, "pair_id": None if is_query else item.pair_id,
                "query_id": item.query_id if is_query else None, "method_id": method_id,
                "score_raw": d.score_raw, "score_direction": d.score_direction.value, "y_pred": d.y_pred,
                "q_valid": d.outcome.value == "decided", "reason_code": d.reason_code, "elapsed_ms": item_ms,
                "outcome": d.outcome.value, "timing_mode": mode,
                **{f"{COMPONENT_PREFIX}{k}": v for k, v in d.components.items()},
            })
        return rows, sum(ms for _, ms, _ in timed_decisions)

    @staticmethod
    def _to_frame(rows: list[dict]) -> pd.DataFrame:
        comps = sorted({k for r in rows for k in r if k.startswith(COMPONENT_PREFIX)})
        return pd.DataFrame(rows, columns=list(PREDICTIONS_COLUMNS) + comps)
