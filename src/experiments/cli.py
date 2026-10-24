"""python -m experiments build-manifests|validate-data|plan|run|aggregate|report --config …"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml

from expcore.cli_common import add_common_args, path_overrides, run_cli
from expcore.config import Config, load_config, register_params_schema
from expcore.errors import ConfigError, DataValidationError
from expcore.manifests import build_manifests
from experiments.validate import report_path, validate_data

MODULE = "experiments"


@dataclass(frozen=True)
class ExperimentsParams:
    """No parameters of its own: experiments orchestrates the module runs."""


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


HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "build-manifests": _build_manifests,
    "validate-data": _validate_data,
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
