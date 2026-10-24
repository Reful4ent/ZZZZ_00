"""Run identity, run.json lifecycle, environment snapshot and the structured log (spec §5.5, §7.3, §7.6, §9)."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from expcore.errors import ProtocolViolationError, ResultTamperedError
from expcore.hashing import sha256_file
from expcore.io import dump_json, load_json
from expcore.schemas import validate_run_json

SCHEMA_VERSION = "1"
ENV_VARS = ("PYTHONHASHSEED", "TF_DETERMINISTIC_OPS", "CUDA_VISIBLE_DEVICES")
# Fields that define the run: a resume under the same run_id must agree on all of them.
_IDENTITY = ("run_id", "config_hash", "dataset_manifest_hash", "seed", "profile", "module", "command",
             "protocol_version", "oracle_mode")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def make_run_id(module: str, profile: str, config_hash: str, seed: int) -> str:
    """D-08: the same configuration and seed is the same run."""
    return f"{module}-{profile}-{config_hash[:12]}-s{seed}"


def _git(repo: Path, *args: str) -> bytes | None:
    try:
        res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    except FileNotFoundError:
        return None
    return res.stdout if res.returncode == 0 else None


def git_state(repo: Path) -> tuple[str | None, str | None]:
    """(HEAD commit, SHA-256 of `git diff HEAD` + untracked files) — (None, None) outside a repository."""
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    if head is None:
        return None, None
    commit = head.decode().strip()
    if not (_git(repo, "status", "--porcelain") or b"").strip():
        return commit, None
    h = hashlib.sha256(_git(repo, "diff", "HEAD", "--binary") or b"")
    untracked = (_git(repo, "ls-files", "--others", "--exclude-standard", "-z") or b"").split(b"\0")
    for name in sorted(n for n in untracked if n):
        h.update(b"\0" + name + b"\0" + (Path(repo) / name.decode()).read_bytes())
    return commit, h.hexdigest()


def collect_environment(*, deterministic_ops: bool | None, extra: Mapping | None = None) -> dict:
    """environment.json content: interpreter, OS, hardware, library versions, determinism notes."""
    libraries = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
                 if d.metadata["Name"]}
    notes = ["opencv_rng_state_not_readable"]
    if deterministic_ops is False:
        notes.append("deterministic_ops_unavailable")
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**20
    except ImportError:
        ram = None
        notes.append("ram_unknown_psutil_unavailable")
    gpu = None
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        gpu = [d.name for d in tf.config.list_physical_devices("GPU")] or None
    else:
        notes.append("gpu_not_probed_tensorflow_not_loaded")
    env = {
        "python": sys.version.split()[0], "implementation": platform.python_implementation(),
        "os": platform.platform(), "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "ram_total_mb": ram, "gpu": gpu, "libraries": dict(sorted(libraries.items())),
        "env_vars": {k: os.environ.get(k) for k in ENV_VARS}, "deterministic_ops": deterministic_ops,
        "nondeterminism_notes": notes, "library_primitives": [],
    }
    env.update(extra or {})
    return env


class JsonlLogger:
    """run.log.jsonl: one JSON object per line with ts_utc, level, run_id, event, payload."""

    def __init__(self, path: Path, run_id: str) -> None:
        self._path = Path(path)
        self._run_id = run_id

    def log(self, level: str, event: str, **payload: Any) -> None:
        line = json.dumps({"ts_utc": now_utc(), "level": level, "run_id": self._run_id, "event": event,
                           "payload": payload}, allow_nan=False, ensure_ascii=False)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class RunRecord:
    """Owns run.json (§5.5) and environment.json; a failed run resumes under the same run_id (§7.6.6)."""

    def __init__(self, run_dir: Path, *, run_id: str, config_hash: str, dataset_manifest_hash: str, seed: int,
                 profile: str, module: str, command: str, protocol_version: str, oracle_mode: bool,
                 post_test_study: bool, git_commit: str | None, dirty_diff_hash: str | None,
                 environment: Mapping[str, Any], library_primitives: Sequence[Mapping[str, Any]],
                 paths_resolved: Mapping[str, str] | None = None) -> None:
        if git_commit is None and profile != "smoke":
            raise ProtocolViolationError(f"profile {profile}: a git commit is required (null only in smoke)")
        self.run_dir = Path(run_dir)
        self._env = dict(environment)
        self._static = {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "config_hash": config_hash,
            "git_commit": git_commit, "dirty_diff_hash": dirty_diff_hash,
            "dataset_manifest_hash": dataset_manifest_hash, "seed": seed, "profile": profile, "module": module,
            "command": command, "protocol_version": protocol_version, "oracle_mode": oracle_mode,
            "post_test_study": post_test_study, "library_primitives": [dict(p) for p in library_primitives],
        }
        if paths_resolved is not None:
            self._static["paths_resolved"] = {k: str(v) for k, v in paths_resolved.items()}
        self._doc: dict | None = None

    @property
    def path(self) -> Path:
        return self.run_dir / "run.json"

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        now = now_utc()
        attempts, started_at = [], now
        if self.path.exists():
            prev = load_json(self.path)
            changed = [k for k in _IDENTITY if prev.get(k) != self._static[k]]
            if changed:
                raise ResultTamperedError(f"{self.path}: existing run differs in {changed}; not resuming")
            if prev["status"] == "completed":
                raise ProtocolViolationError(f"{self.path}: run is completed; results are never recomputed")
            attempts, started_at = prev["attempts"], prev["started_at"]
            if attempts and attempts[-1]["status"] == "running":   # previous process died without finishing
                attempts[-1]["status"] = "interrupted"
        attempts.append({"started_at": now, "finished_at": None, "status": "running", "failure": None})
        env_path = self.run_dir / "environment.json"
        dump_json(self._env, env_path)
        environment = {"path": env_path.name, "sha256": sha256_file(env_path),
                       **{k: self._env.get(k) for k in ("python", "os", "cpu", "gpu")}}
        self._doc = {**self._static, "environment": environment, "status": "running", "started_at": started_at,
                     "finished_at": None, "attempts": attempts, "failure": None}
        self._write()

    def finish(self, status: str, failure: Mapping[str, Any] | None = None) -> None:
        if self._doc is None:
            raise ProtocolViolationError("RunRecord.finish() before start()")
        if status not in ("completed", "failed"):
            raise ValueError(f"finish status must be completed or failed, got {status!r}")
        now = now_utc()
        failure = dict(failure) if failure is not None else None
        self._doc.update(status=status, finished_at=now, failure=failure)
        self._doc["attempts"][-1].update(finished_at=now, status=status, failure=failure)
        self._write()

    def _write(self) -> None:
        validate_run_json(self._doc)
        dump_json(self._doc, self.path)
