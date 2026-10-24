"""Preparation of a metric run: the same steps and checks as the preparation part of
expcore.cli_common.execute_run (spec 3.4 item 4), plus metric's profile validation, thread limits and stage runs.

Stage runs (Phase 7 ruling, TZ00 D-08): `train` is the base run of its config; `embed`/`search` are separate
runs whose config_hash = sha256(canonical_json({"base_config_hash", "stage"})). TF-free (TF threads are set
only when the command has already imported TensorFlow).
"""
import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import yaml

from expcore.cli_common import path_overrides, plan_dir
from expcore.config import Config, config_hash, load_config
from expcore.errors import ConfigError, ProtocolViolationError
from expcore.hashing import canonical_json, sha256_text
from expcore.io import atomic_write_bytes, dump_json, load_json
from expcore.loaders import ImageStore, SplitGuard, TEST_SPLITS
from expcore.manifests import LoadedManifest, load_manifest, manifest_version
from expcore.primitives import primitive_versions
from expcore.protocol import check_run_in_plan, is_post_test, open_test
from expcore.resume import SUCCESS, _resolved_config
from expcore.runlog import RunRecord, collect_environment, git_state, make_run_id
from expcore.seeding import capture_rng_state, init_seeds
from metric.data import check_disjoint
from metric.params import params_dict, validate_profile

MODULE = "metric"
NONDETERMINISM_NOTES = (
    "degrade_stream_fallback: query degradations draw from stream(seed, 'augment', 2**31 + i) until TZ00 adds "
    "a 'degrade' stream (spec 3.4 item 1)",
    "tf_op_determinism: results are reproducible on the same device type and library versions; CPU and GPU "
    "runs are not bitwise comparable",
)


@dataclass
class MetricRun:
    cfg: Config
    method_id: str
    run_id: str
    run_dir: Path
    manifest: LoadedManifest
    guard: SplitGuard
    store: ImageStore
    record: RunRecord
    registry_path: Path
    plan: Path
    config_hash: str        # of this run (the stage hash for embed/search)
    base_config_hash: str   # of the config file
    params: dict
    images: pd.DataFrame    # SOP rows of the frozen manifest
    stage: dict | None = None

    def open_split(self, split: str) -> None:
        """Before the first image of `split` is loaded: confirmatory opens test through the protocol (7.5)."""
        if self.cfg.profile == "confirmatory" and split in TEST_SPLITS:
            open_test(self.registry_path, self.plan, self.cfg.protocol_version, self.run_id)
            self.guard.allow_test()
        self.guard.check(split)

    def write_run_files(self) -> None:
        """config.resolved.yaml (the base config, as expcore's Runner writes it), stage.json, rng_state.json."""
        if not (self.run_dir / "config.resolved.yaml").exists():
            atomic_write_bytes(self.run_dir / "config.resolved.yaml", yaml.safe_dump(
                _resolved_config(self.cfg), allow_unicode=True, sort_keys=True).encode("utf-8"))
        if self.stage is not None and not (self.run_dir / "stage.json").exists():
            dump_json({"base_config_hash": self.base_config_hash, "stage": self.stage,
                       "base_run_id": source_run_id(self.cfg)}, self.run_dir / "stage.json")
        if not (self.run_dir / "rng_state.json").exists():
            dump_json({"start": capture_rng_state()}, self.run_dir / "rng_state.json")


def stage_hash(base_hash: str, stage: dict) -> str:
    return sha256_text(canonical_json({"base_config_hash": base_hash, "stage": stage}))


def source_run_id(cfg: Config) -> str:
    """run_id of the train run whose encoder a stage uses: the same config without the query degradation (E7)."""
    src = dataclasses.replace(cfg, params=dataclasses.replace(cfg.params, degradation="none"))
    return make_run_id(MODULE, cfg.profile, config_hash(src), cfg.seeds.seed)


def completed_source(cfg: Config) -> Path:
    run_dir = Path(cfg.paths.out_root) / cfg.profile / MODULE / source_run_id(cfg)
    run_json = run_dir / "run.json"
    if not ((run_dir / SUCCESS).exists() and run_json.exists() and load_json(run_json)["status"] == "completed"):
        raise ProtocolViolationError(f"{run_dir}: the train run of this config is not completed; train it first")
    return run_dir


def _set_tf_threads(n: int) -> dict | None:
    tf = sys.modules.get("tensorflow")
    if tf is None:
        return None
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(n)
    except RuntimeError:   # TF runtime already initialised in this process (e.g. several commands in one test)
        pass
    return {"intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op": tf.config.threading.get_inter_op_parallelism_threads()}


def prepare_run(command: str, args: argparse.Namespace, *, deterministic_ops: bool | None,
                stage: dict | None = None) -> MetricRun:
    """config → frozen manifest → profile/split checks → seeds → run dir → RunRecord (not started)."""
    cfg = load_config(args.config, module=MODULE, path_overrides=path_overrides(args))
    if len(cfg.algorithms) != 1:
        raise ConfigError(f"algorithms: a {MODULE} {command} runs exactly one method, got {sorted(cfg.algorithms)}")
    validate_profile(cfg)
    (base_method,) = cfg.algorithms
    oracle = cfg.data_modes.oracle.enabled
    method_id = f"{base_method}+oracle" if oracle else base_method
    seed = cfg.seeds.seed
    if seed is None:
        raise ConfigError("seeds.seed: a single run needs a seed")
    version = manifest_version(cfg)
    manifest = load_manifest(cfg.paths.manifest_dir, version)
    images = manifest.images[manifest.images["dataset_id"] == "sop"].reset_index(drop=True)
    check_disjoint(images)   # spec 4.2: LeakageError (exit 3)
    fixture = all(ds.fixture for ds in cfg.datasets.values())
    guard = SplitGuard(cfg.profile, fixture=fixture)
    base_h = config_hash(cfg)
    h = base_h if stage is None else stage_hash(base_h, stage)
    run_id = make_run_id(MODULE, cfg.profile, h, seed)
    plan = plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version)
    registry_path = Path(args.protocol_registry)
    if cfg.profile == "confirmatory":
        if not registry_path.is_file():
            raise ProtocolViolationError(f"{registry_path}: protocol registry is required for confirmatory runs")
        check_run_in_plan(plan, base_h)   # the plan lists the config; its stages run under it

    init_seeds(seed)
    cv2.setNumThreads(cfg.determinism.opencv_threads)
    tf_threads = _set_tf_threads(cfg.determinism.numpy_threads or 1)
    run_dir = cfg.paths.out_root / cfg.profile / MODULE / run_id
    previous = run_dir / "run.json"
    if previous.exists() and load_json(previous)["command"] != command:   # D-08 run_id carries no command
        raise ProtocolViolationError(f"{run_dir} belongs to `{MODULE} {load_json(previous)['command']}`; "
                                     f"`{command}` needs its own config")
    env = collect_environment(deterministic_ops=deterministic_ops,
                              extra=None if tf_threads is None else {"tf_threads": tf_threads})
    env["nondeterminism_notes"] = [*env["nondeterminism_notes"], *NONDETERMINISM_NOTES]
    commit, dirty = git_state(Path.cwd())
    record = RunRecord(
        run_dir, run_id=run_id, config_hash=h, dataset_manifest_hash=manifest.manifest_hash, seed=seed,
        profile=cfg.profile, module=MODULE, command=command, protocol_version=cfg.protocol_version,
        oracle_mode=oracle,
        post_test_study=registry_path.is_file() and is_post_test(registry_path, cfg.protocol_version),
        git_commit=commit, dirty_diff_hash=dirty, environment=env,
        library_primitives=primitive_versions(), paths_resolved={"data_root": cfg.paths.data_root,
                                               "manifest_dir": cfg.paths.manifest_dir, "out_root": cfg.paths.out_root})
    return MetricRun(cfg, method_id, run_id, run_dir, manifest, guard, ImageStore(cfg.paths.data_root, images, guard),
                     record, registry_path, plan, h, base_h, params_dict(cfg.params), images, stage)
