"""Strict config loader (spec §5.6): every key required, unknown keys rejected, no defaults."""

import collections.abc
import dataclasses
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from expcore.errors import ConfigError
from expcore.hashing import canonical_json, sha256_text

ALLOWED_SEEDS = {"smoke": (11,), "pilot": (11, 23, 37), "confirmatory": (11, 23, 37, 51, 71)}
DATASET_IDS = ("kadid10k", "hpatches", "mvtec_ad", "sop")
KADID_N_VALIDATION = 20
REQUIRED_PAIR_COLUMNS = ("pair_id", "query_id", "reference_id", "relation_id")
OPTIONAL_PAIR_COLUMNS = ("transform_id", "severity", "seed", "input_mask_path")
FORBIDDEN_PAIR_COLUMNS = ("y_match", "gt_H_path", "gt_mask_path")
COLOR_ORDERS = ("bgr", "rgb", "gray")
TIE_RULES = (">=", ">")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class SeedsCfg:
    seed: int | None
    seeds: tuple[int, ...] | None


@dataclass(frozen=True)
class DatasetCfg:
    dataset_id: str
    manifest_version: str
    fixture: bool
    categories: tuple[str, ...] | None
    n_validation: int | None
    expected: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class InputsCfg:
    pair_columns: tuple[str, ...]
    color_order: str


@dataclass(frozen=True)
class OracleCfg:
    enabled: bool
    reason: str | None


@dataclass(frozen=True)
class DataModesCfg:
    oracle: OracleCfg


@dataclass(frozen=True)
class DecisionCfg:
    tie_rule: str
    criterion_id: str | None


@dataclass(frozen=True)
class DeterminismCfg:
    deterministic_ops: bool
    opencv_threads: int
    numpy_threads: int


@dataclass(frozen=True)
class RunnerCfg:
    chunk_size: int


@dataclass(frozen=True)
class LoggingCfg:
    level: str
    dir: str


@dataclass(frozen=True)
class PathsCfg:
    data_root: Path
    manifest_dir: Path
    out_root: Path


@dataclass(frozen=True)
class Config:
    schema_version: str
    protocol_version: str
    profile: str
    seeds: SeedsCfg
    datasets: Mapping[str, DatasetCfg]
    inputs: InputsCfg
    data_modes: DataModesCfg
    algorithms: Mapping[str, str]
    params: Any  # replaced by the module's registered schema instance
    decision: DecisionCfg
    determinism: DeterminismCfg
    runner: RunnerCfg
    logging: LoggingCfg
    paths: PathsCfg


_PARAMS_SCHEMAS: dict[str, type] = {}


def register_params_schema(module: str, schema: type) -> None:
    if not dataclasses.is_dataclass(schema):
        raise TypeError(f"params schema for {module!r} must be a dataclass")
    _PARAMS_SCHEMAS[module] = schema


def _plain(value: Any, where: str) -> Any:
    """Free-form values must be JSON-representable so that config_hash is defined."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where}: NaN/Inf not allowed")
        return value
    if isinstance(value, list):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise ConfigError(f"{where}: key {k!r} must be a string")
        return {k: _plain(v, f"{where}.{k}") for k, v in value.items()}
    raise ConfigError(f"{where}: unsupported value of type {type(value).__name__} (quote it as a string)")


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if tp is Any:
        return _plain(value, where)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: must not be null")
        (inner,) = [a for a in args if a is not type(None)]
        return _convert(inner, value, where)
    if value is None:
        raise ConfigError(f"{where}: must not be null")
    if dataclasses.is_dataclass(tp):
        return _from_dict(tp, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        (item,) = [a for a in typing.get_args(tp) if a is not Ellipsis]
        return tuple(_convert(item, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (collections.abc.Mapping, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        _, val_tp = typing.get_args(tp)
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{where}: key {k!r} must be a string")
            out[k] = _convert(val_tp, v, f"{where}.{k}")
        return out
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    elif tp is Path:
        if isinstance(value, (str, Path)):
            return Path(value)
    else:
        raise TypeError(f"{where}: unsupported schema type {tp!r}")
    raise ConfigError(f"{where}: expected {tp.__name__}, got {type(value).__name__} {value!r}")


def _from_dict(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or '<root>'}: expected a mapping")
    prefix = f"{where}." if where else ""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(str(k) for k in data if k not in names)
    if unknown:
        raise ConfigError(f"unknown key(s): {', '.join(prefix + k for k in unknown)}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"missing key(s): {', '.join(prefix + n for n in missing)}")
    return cls(**{n: _convert(hints[n], data[n], prefix + n) for n in names})


def strict_from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    return _from_dict(cls, data, "")


def _validate(cfg: Config) -> None:
    if cfg.profile not in ALLOWED_SEEDS:
        raise ConfigError(f"profile: {cfg.profile!r} not in {sorted(ALLOWED_SEEDS)}")
    allowed = ALLOWED_SEEDS[cfg.profile]
    if cfg.seeds.seed is not None and cfg.seeds.seed not in allowed:
        raise ConfigError(f"seeds.seed: {cfg.seeds.seed} not allowed for profile {cfg.profile} {allowed}")
    if cfg.seeds.seeds is not None:
        bad = [s for s in cfg.seeds.seeds if s not in allowed]
        if bad or len(set(cfg.seeds.seeds)) != len(cfg.seeds.seeds):
            raise ConfigError(f"seeds.seeds: {list(cfg.seeds.seeds)} must be distinct values from {allowed}")

    for key, ds in cfg.datasets.items():
        where = f"datasets.{key}"
        if ds.dataset_id != key or key not in DATASET_IDS:
            raise ConfigError(f"{where}.dataset_id: {ds.dataset_id!r} must equal its key and be one of {DATASET_IDS}")
        if ds.fixture != (cfg.profile == "smoke"):
            raise ConfigError(f"{where}.fixture: the smoke profile accepts only fixture datasets"
                              if cfg.profile == "smoke" else
                              f"{where}.fixture: fixture datasets are allowed only in the smoke profile")
        if key == "kadid10k":
            if ds.n_validation is None or ds.n_validation < 1:
                raise ConfigError(f"{where}.n_validation: positive integer required")
            if not ds.fixture and ds.n_validation != KADID_N_VALIDATION:
                raise ConfigError(f"{where}.n_validation: real KADID requires {KADID_N_VALIDATION}")

    cols = cfg.inputs.pair_columns
    leaked = [c for c in cols if c in FORBIDDEN_PAIR_COLUMNS]
    if leaked:
        raise ConfigError(f"inputs.pair_columns: evaluation column(s) {leaked} are not visible to algorithms")
    unknown = [c for c in cols if c not in REQUIRED_PAIR_COLUMNS + OPTIONAL_PAIR_COLUMNS]
    missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in cols]
    if unknown or missing or len(set(cols)) != len(cols):
        raise ConfigError(f"inputs.pair_columns: unknown {unknown}, missing {missing} or duplicates in {list(cols)}")
    if cfg.inputs.color_order not in COLOR_ORDERS:
        raise ConfigError(f"inputs.color_order: {cfg.inputs.color_order!r} not in {COLOR_ORDERS}")

    oracle = cfg.data_modes.oracle
    if oracle.enabled and not (oracle.reason or "").strip():
        raise ConfigError("data_modes.oracle.reason: required when oracle is enabled")
    if cfg.decision.tie_rule not in TIE_RULES:
        raise ConfigError(f"decision.tie_rule: {cfg.decision.tie_rule!r} not in {TIE_RULES}")
    for name in ("opencv_threads", "numpy_threads"):
        if getattr(cfg.determinism, name) < 1:
            raise ConfigError(f"determinism.{name}: must be >= 1")
    if cfg.runner.chunk_size < 1:
        raise ConfigError("runner.chunk_size: must be >= 1")
    if cfg.logging.level not in LOG_LEVELS:
        raise ConfigError(f"logging.level: {cfg.logging.level!r} not in {LOG_LEVELS}")


def load_config(path: Path, *, module: str, path_overrides: Mapping[str, Path] | None = None) -> Config:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: cannot read config: {e}") from e
    cfg = strict_from_dict(Config, data)
    schema = _PARAMS_SCHEMAS.get(module)
    if schema is None:
        raise ConfigError(f"params: no schema registered for module {module!r}")
    cfg = dataclasses.replace(cfg, params=_from_dict(schema, cfg.params, "params"))
    if path_overrides:
        valid = [f.name for f in dataclasses.fields(PathsCfg)]
        bad = sorted(k for k in path_overrides if k not in valid)
        if bad:
            raise ConfigError(f"path override(s) {bad} not in {valid}")
        cfg = dataclasses.replace(cfg, paths=dataclasses.replace(
            cfg.paths, **{k: Path(v) for k, v in path_overrides.items()}))
    _validate(cfg)
    return cfg


def config_hash(cfg: Config) -> str:
    """SHA-256 of the canonical JSON of the config without the `paths` section (D-09)."""
    d = dataclasses.asdict(cfg)
    del d["paths"]
    return sha256_text(canonical_json(d))
