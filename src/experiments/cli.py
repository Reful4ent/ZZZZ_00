"""python -m experiments build-manifests|validate-data|plan|run|aggregate|report --config …"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml

from expcore.cli_common import add_common_args, path_overrides, plan_dir, run_cli
from expcore.config import Config, load_config, register_params_schema
from expcore.errors import ConfigError, DataValidationError
from expcore.io import atomic_write_bytes, load_json, read_parquet
from expcore.manifests import build_manifests, load_manifest, manifest_version
from experiments.aggregate import aggregate, run_statuses
from experiments.plan import PlanEntry, load_plan, make_plan, run_plan, write_plan
from experiments.report import render_report
from experiments.validate import report_path, validate_data

MODULE = "experiments"


@dataclass(frozen=True)
class ExperimentsParams:
    """The run matrix: module configs that `plan` expands over the profile's seeds."""
    runs: tuple[PlanEntry, ...]


register_params_schema(MODULE, ExperimentsParams)


def _config(args: argparse.Namespace) -> Config:
    return load_config(args.config, module=MODULE, path_overrides=path_overrides(args))


def _build_manifests(args: argparse.Namespace) -> None:
    cfg = _config(args)
    overrides = None
    if args.overrides is not None:
        try:
            overrides = yaml.safe_load(Path(args.overrides).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"{args.overrides}: cannot read overrides: {e}") from e
    info = build_manifests(cfg, cfg.paths.data_root, cfg.paths.manifest_dir, overrides)
    print(f"manifest {info.version} frozen: {info.manifest_hash}")


def _validate_data(args: argparse.Namespace) -> None:
    cfg = _config(args)
    report = validate_data(cfg, cfg.paths.data_root, cfg.paths.manifest_dir)
    if not report["ok"]:
        failed = ", ".join(i["check"] for i in report["issues"])
        raise DataValidationError(f"validate-data failed ({failed}); see {report_path(cfg)}")
    print(f"manifest {report['version']} valid")


def _plan(args: argparse.Namespace) -> None:
    cfg = _config(args)
    out = plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version)
    runs = write_plan(out, make_plan(cfg, Path(args.config).parent), cfg, registry_path=args.protocol_registry)
    print(f"{len(runs)} runs planned in {out}")


def _run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    extra = ["--protocol-registry", str(args.protocol_registry)]
    for flag, value in (("--data-root", args.data_root), ("--manifest-dir", args.manifest_dir),
                        ("--out-root", args.out_root)):
        if value is not None:
            extra += [flag, str(value)]
    return run_plan(plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version), extra)


def _stage_path(cfg: Config, stage: str, name: str) -> Path:
    return cfg.paths.out_root / cfg.profile / stage / cfg.protocol_version / name


def _aggregate(args: argparse.Namespace) -> None:
    cfg = _config(args)
    out = _stage_path(cfg, "aggregate", "metrics.parquet")
    df = aggregate(plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version), out)
    print(f"{df['run_id'].nunique()} completed runs, {len(df)} metric rows -> {out}")


def _report(args: argparse.Namespace) -> None:
    cfg = _config(args)
    metrics_path = _stage_path(cfg, "aggregate", "metrics.parquet")
    if not metrics_path.is_file():
        raise DataValidationError(f"{metrics_path} is missing: run `experiments aggregate` first")
    pdir = plan_dir(cfg.paths.out_root, cfg.profile, cfg.protocol_version)
    manifest = load_manifest(cfg.paths.manifest_dir, manifest_version(cfg))
    # completeness from datasets.json, which the D-07 manifest hash covers (completeness_report.json is not)
    datasets = {rec["dataset_id"]: rec for rec in manifest.datasets}
    completeness = {"runs": run_statuses(pdir), "datasets": {
        d: ({"status": datasets[d]["completeness"], "subset_name": datasets[d].get("subset_name")}
            if datasets.get(d, {}).get("completeness") else None) for d in sorted(cfg.datasets)}}
    text = render_report(read_parquet(metrics_path), load_plan(pdir), completeness)
    out = _stage_path(cfg, "report", "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out, text.encode("utf-8"))
    print(f"report -> {out}")


HANDLERS: dict[str, Callable[[argparse.Namespace], int | None]] = {
    "build-manifests": _build_manifests,
    "validate-data": _validate_data,
    "plan": _plan,
    "run": _run,
    "aggregate": _aggregate,
    "report": _report,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=MODULE)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in HANDLERS:
        p = add_common_args(sub.add_parser(command))
        if command == "build-manifests":
            p.add_argument("--overrides", type=Path, help="overrides.yaml with exclude_image: [{sha256, reason}]")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(lambda args: HANDLERS[args.command](args), argv, parser=build_parser())
