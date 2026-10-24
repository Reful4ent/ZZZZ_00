"""experiments plan / run (spec §7.1, §7.5): expand module configs over the profile's seeds, then execute."""

import dataclasses
import importlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from expcore.config import Config, _UniqueKeyLoader, config_hash, load_config
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


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)   # same strictness as load_config
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: cannot read module config: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: a module config must be a mapping")
    return data


def make_plan(cfg: Config, base_dir: Path) -> list[tuple[PlannedRun, str]]:
    """One run per (entry, seed in cfg.seeds.seeds), with the per-seed module config text; nothing is written.

    config_hash includes the seed, so every seed gets its own expanded config.
    """
    if not cfg.seeds.seeds:
        raise ConfigError("seeds.seeds: a plan needs the profile's seeds")
    # Expanded configs take the plan's resolved paths (CLI overrides included); paths are outside config_hash.
    plan_paths = {k: str(v) for k, v in dataclasses.asdict(cfg.paths).items()}
    planned: list[tuple[PlannedRun, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, entry in enumerate(cfg.params.runs):
            if entry.command not in MODULE_COMMANDS.get(entry.module, ()):
                raise ConfigError(f"params.runs[{i}]: unknown {entry.module} {entry.command}")
            importlib.import_module(f"{entry.module}.cli")   # registers the module's params schema
            source = Path(base_dir) / entry.config
            module_cfg = _read_yaml(source)
            for key in ("profile", "protocol_version"):
                if module_cfg.get(key) != getattr(cfg, key):
                    raise ConfigError(f"{source}: {key} {module_cfg.get(key)!r} differs from the plan's "
                                      f"{getattr(cfg, key)!r}")
            for seed in cfg.seeds.seeds:
                text = yaml.safe_dump({**module_cfg, "seeds": {"seed": seed, "seeds": None}, "paths": plan_paths},
                                      allow_unicode=True, sort_keys=True)
                probe = Path(tmp) / f"{i}-{seed}.yaml"
                probe.write_text(text, encoding="utf-8")
                h = config_hash(load_config(probe, module=entry.module))   # validates the expanded config
                rel = Path("configs") / f"{i:02d}-{entry.module}-{entry.command}-{Path(entry.config).stem}-s{seed}.yaml"
                planned.append((PlannedRun(entry.module, entry.command, seed, rel.as_posix(), h,
                                           make_run_id(entry.module, cfg.profile, h, seed)), text))
    run_ids = [r.run_id for r, _ in planned]
    dup = sorted({x for x in run_ids if run_ids.count(x) > 1})
    if dup:
        raise ConfigError(f"params.runs: entries share run_ids {dup} (D-08 run_id = module, profile, config, seed — "
                          "give each command its own config)")
    return planned


def write_plan(out_dir: Path, planned: Sequence[tuple[PlannedRun, str]], cfg: Config, *,
               registry_path: Path) -> list[PlannedRun]:
    """Lock first (confirmatory; refused once test is open and the plan changes), then configs and plan.json."""
    out_dir = Path(out_dir)
    runs = [r for r, _ in planned]
    if cfg.profile == "confirmatory":
        if is_post_test(registry_path, cfg.protocol_version):
            print(f"warning: protocol {cfg.protocol_version} descends from a version with an open test - "
                  "results are a separate post-test study", file=sys.stderr)
        write_plan_lock(out_dir, cfg.protocol_version, [dataclasses.asdict(r) for r in runs],
                        registry_path=registry_path)
    for r, text in planned:
        path = out_dir / r.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, text.encode("utf-8"))
    dump_json({"protocol_version": cfg.protocol_version, "profile": cfg.profile, "created_at": now_utc(),
               "runs": [dataclasses.asdict(r) for r in runs]}, out_dir / "plan.json")
    return runs


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
