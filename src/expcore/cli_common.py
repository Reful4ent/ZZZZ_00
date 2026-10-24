"""Shared CLI plumbing (spec §8): arguments, exit codes (§8.3) and the module `run` pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2

from expcore.config import config_hash, load_config
from expcore.errors import ConfigError, DataValidationError, NotImplementedAlgorithm, ProtocolError, \
    ProtocolViolationError
from expcore.loaders import ImageStore, QueryInput, SplitGuard, TEST_SPLITS, load_pair_inputs
from expcore.manifests import load_manifest, manifest_version
from expcore.protocol import check_run_in_plan, is_post_test, open_test
from expcore.registry import get_algorithm
from expcore.resume import SUCCESS, Runner, RunContext
from expcore.runlog import JsonlLogger, RunRecord, collect_environment, git_state, make_run_id
from expcore.seeding import init_seeds

DEFAULT_REGISTRY = Path("configs") / "protocol_registry.json"
PROGRAM_OF = {"photometric": "p1", "geometric": "p2", "metric": "p3"}
TECHNICAL_EXIT = 4


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=Path, required=True, help="config.yaml (profile, seed, params live here)")
    p.add_argument("--data-root", type=Path, help="override paths.data_root")
    p.add_argument("--manifest-dir", type=Path, help="override paths.manifest_dir")
    p.add_argument("--out-root", type=Path, help="override paths.out_root")
    p.add_argument("--protocol-registry", type=Path, default=DEFAULT_REGISTRY,
                   help=f"protocol version registry (default {DEFAULT_REGISTRY.as_posix()})")
    return p


def base_parser(prog: str) -> argparse.ArgumentParser:
    return add_common_args(argparse.ArgumentParser(prog=prog))


def command_parser(prog: str, commands: Sequence[str]) -> argparse.ArgumentParser:
    """`prog <command> --config …` with the common arguments on every command."""
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in commands:
        add_common_args(sub.add_parser(command))
    return parser


def run_cli(main: Callable[[argparse.Namespace], None], argv: Sequence[str] | None, *,
            parser: argparse.ArgumentParser) -> int:
    """Parse, run, map exceptions to exit codes: ProtocolError → its code, anything else → 4."""
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:   # --help (0) or bad arguments (2)
        return e.code if isinstance(e.code, int) else 2
    try:
        main(args)
        return 0
    except ProtocolError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return e.exit_code
    except Exception as e:
        print(f"TechnicalError: {type(e).__name__}: {e}", file=sys.stderr)
        return TECHNICAL_EXIT


def path_overrides(args: argparse.Namespace) -> dict[str, Path]:
    pairs = (("data_root", args.data_root), ("manifest_dir", args.manifest_dir), ("out_root", args.out_root))
    return {k: v for k, v in pairs if v is not None}


def plan_dir(out_root: Path, profile: str, protocol_version: str) -> Path:
    return Path(out_root) / profile / "plan" / protocol_version


def execute_run(module: str, command: str, args: argparse.Namespace, *, query: bool,
                deterministic_ops: bool | None = None) -> None:
    """config → frozen manifest → profile/split checks → seeds → run dir → RunRecord → algorithm → Runner."""
    cfg = load_config(args.config, module=module, path_overrides=path_overrides(args))
    if len(cfg.algorithms) != 1:
        raise ConfigError(f"algorithms: a {module} {command} runs exactly one method, got {sorted(cfg.algorithms)}")
    (base_method,) = cfg.algorithms
    oracle = cfg.data_modes.oracle.enabled
    method_id = f"{base_method}+oracle" if oracle else base_method   # §5.3: oracle runs are marked everywhere
    seed = cfg.seeds.seed
    if seed is None:
        raise ConfigError("seeds.seed: a single run needs a seed")
    version = manifest_version(cfg)
    manifest = load_manifest(cfg.paths.manifest_dir, version)
    guard = SplitGuard(cfg.profile, fixture=all(ds.fixture for ds in cfg.datasets.values()))
    h = config_hash(cfg)
    run_id = make_run_id(module, cfg.profile, h, seed)
    plan = plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version)
    registry_path = Path(args.protocol_registry)
    if cfg.profile == "confirmatory":
        if not registry_path.is_file():
            raise ProtocolViolationError(f"{registry_path}: protocol registry is required for confirmatory runs")
        check_run_in_plan(plan, h)

    init_seeds(seed)
    cv2.setNumThreads(cfg.determinism.opencv_threads)
    run_dir = cfg.paths.out_root / cfg.profile / module / run_id
    commit, dirty = git_state(Path.cwd())
    record = RunRecord(
        run_dir, run_id=run_id, config_hash=h, dataset_manifest_hash=manifest.manifest_hash, seed=seed,
        profile=cfg.profile, module=module, command=command, protocol_version=cfg.protocol_version,
        oracle_mode=oracle,
        post_test_study=registry_path.is_file() and is_post_test(registry_path, cfg.protocol_version),
        git_commit=commit, dirty_diff_hash=dirty, environment=collect_environment(deterministic_ops=deterministic_ops),
        library_primitives=[], paths_resolved={"data_root": cfg.paths.data_root,
                                               "manifest_dir": cfg.paths.manifest_dir, "out_root": cfg.paths.out_root})
    try:
        predictor = get_algorithm(module, base_method)
    except NotImplementedAlgorithm as e:
        if not (run_dir / SUCCESS).exists():
            failure = {"type": "NotImplementedAlgorithm", "message": str(e), "unit_id": None}
            record.start()
            JsonlLogger(run_dir / "errors.jsonl", run_id).log("ERROR", "run_failed", **failure)
            record.finish("failed", failure)
        raise

    program = PROGRAM_OF[module]
    pairs = manifest.pairs.get(program)
    if pairs is None:
        raise DataValidationError(f"manifest {version!r} has no {program}/pairs.parquet (no pair builder registered)")
    splits = sorted(set(pairs["split"]))
    if cfg.profile == "confirmatory":
        if TEST_SPLITS & set(splits):   # §7.5: record the opening before any test image is loaded
            open_test(registry_path, plan, cfg.protocol_version, run_id)
            guard.allow_test()
    else:
        splits = [s for s in splits if _allowed(guard, s)]
    pairs = pairs[pairs["split"].isin(splits)]
    if query:
        items = [QueryInput(q, {}) for q in sorted(set(pairs["query_id"]))]
    else:
        items = [p for s in splits for p in load_pair_inputs(pairs, cfg.inputs.pair_columns, s, guard)]
    ctx = RunContext(run_id=run_id, seed=seed, profile=cfg.profile, run_dir=run_dir, cfg=cfg, record=record)
    Runner(ctx, predictor, ImageStore(cfg.paths.data_root, manifest.images, guard)).run(items, method_id=method_id)


def _allowed(guard: SplitGuard, split: str) -> bool:
    try:
        guard.check(split)
        return True
    except ProtocolViolationError:
        return False
