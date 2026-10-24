"""experiments plan / run (spec §7.1, §7.5): expand module configs over the profile's seeds, then execute."""

import dataclasses
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from expcore.config import Config, config_hash, load_config
from expcore.errors import ConfigError, DataValidationError
from expcore.io import atomic_write_bytes, dump_json, load_json
from expcore.protocol import is_post_test, write_plan_lock
from expcore.runlog import make_run_id, now_utc

MODULE_COMMANDS = {"photometric": ("run",), "geometric": ("run",), "metric": ("train", "embed", "search")}


@dataclass(frozen=True)
class PlanEntry:
    module: str
    command: str
    config: str   # module config, relative to the experiments config


@dataclass(frozen=True)
class PlannedRun:
    module: str
    command: str
    seed: int
    config_path: str   # relative to the plan directory
    config_hash: str
    run_id: str


def make_plan(cfg: Config, base_dir: Path, out_dir: Path) -> list[PlannedRun]:
    """One run per (entry, seed in cfg.seeds.seeds); per-seed module configs are written to out_dir/configs/."""
    if not cfg.seeds.seeds:
        raise ConfigError("seeds.seeds: a plan needs the profile's seeds")
    runs = []
    for entry in cfg.params.runs:
        if entry.command not in MODULE_COMMANDS.get(entry.module, ()):
            raise ConfigError(f"params.runs: unknown {entry.module} {entry.command}")
        importlib.import_module(f"{entry.module}.cli")   # registers the module's params schema
        source = Path(base_dir) / entry.config
        try:
            module_cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"{source}: cannot read module config: {e}") from e
        for key in ("profile", "protocol_version"):
            if module_cfg.get(key) != getattr(cfg, key):
                raise ConfigError(f"{source}: {key} {module_cfg.get(key)!r} differs from the plan's "
                                  f"{getattr(cfg, key)!r}")
        for seed in cfg.seeds.seeds:
            d = {**module_cfg, "seeds": {"seed": seed, "seeds": None}}
            rel = Path("configs") / f"{entry.module}-{entry.command}-{Path(entry.config).stem}-s{seed}.yaml"
            path = Path(out_dir) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, yaml.safe_dump(d, allow_unicode=True, sort_keys=True).encode("utf-8"))
            h = config_hash(load_config(path, module=entry.module))   # validates the expanded config
            runs.append(PlannedRun(entry.module, entry.command, seed, rel.as_posix(), h,
                                   make_run_id(entry.module, cfg.profile, h, seed)))
    return runs


def write_plan(out_dir: Path, runs: Sequence[PlannedRun], cfg: Config, *, registry_path: Path) -> None:
    """plan.json; for confirmatory first plan.lock.json (refused when the version's test is open and it changes)."""
    out_dir = Path(out_dir)
    if cfg.profile == "confirmatory":
        if is_post_test(registry_path, cfg.protocol_version):
            print(f"warning: protocol {cfg.protocol_version} descends from a version with an open test — "
                  "results are a separate post-test study", file=sys.stderr)
        write_plan_lock(out_dir, cfg.protocol_version, [dataclasses.asdict(r) for r in runs],
                        registry_path=registry_path)
    dump_json({"protocol_version": cfg.protocol_version, "profile": cfg.profile, "created_at": now_utc(),
               "runs": [dataclasses.asdict(r) for r in runs]}, out_dir / "plan.json")


def load_plan(out_dir: Path) -> list[PlannedRun]:
    path = Path(out_dir) / "plan.json"
    if not path.is_file():
        raise DataValidationError(f"{path} is missing: run `experiments plan` first")
    return [PlannedRun(**r) for r in load_json(path)["runs"]]


def run_plan(out_dir: Path, extra_argv: Sequence[str] = ()) -> int:
    """Execute every planned run through its module's main; the result is the largest exit code."""
    codes = []
    for r in load_plan(out_dir):
        main = importlib.import_module(f"{r.module}.cli").main
        codes.append(main([r.command, "--config", str(Path(out_dir) / r.config_path), *extra_argv]))
    return max(codes, default=0)
