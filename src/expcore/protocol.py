"""Protocol versions and the test lock (spec §7.5, D-16, ВОСП-04).

configs/protocol_registry.json: [{protocol_version, created_at, parent_version, test_opened, plan_hash}].
A confirmatory run must be in plan.lock.json; the first access to test writes test_opened.json and
marks the version; after that its plan cannot change and descendants are post-test studies.
"""

from pathlib import Path
from typing import Mapping, Sequence

from expcore.errors import DataValidationError, ProtocolViolationError
from expcore.hashing import canonical_json, sha256_text
from expcore.io import dump_json, load_json
from expcore.runlog import now_utc

_ENTRY_KEYS = {"protocol_version", "created_at", "parent_version", "test_opened", "plan_hash"}
PLAN_LOCK = "plan.lock.json"
TEST_OPENED = "test_opened.json"


def load_registry(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    entries = load_json(path)
    if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != _ENTRY_KEYS for e in entries):
        raise DataValidationError(f"{path}: every entry needs exactly {sorted(_ENTRY_KEYS)}")
    versions = [e["protocol_version"] for e in entries]
    if len(set(versions)) != len(versions):
        raise DataValidationError(f"{path}: duplicate protocol versions")
    return entries


def _entry(entries: list[dict], version: str, path: Path) -> dict:
    for e in entries:
        if e["protocol_version"] == version:
            return e
    raise ProtocolViolationError(f"{path}: protocol version {version!r} is not registered")


def register_version(path: Path, version: str, parent: str | None) -> None:
    """A changed parameter grid is a new version; existing versions are never edited."""
    entries = load_registry(path)
    if any(e["protocol_version"] == version for e in entries):
        raise ProtocolViolationError(f"{path}: protocol version {version!r} already exists; register a new one")
    if parent is not None:
        _entry(entries, parent, path)
    entries.append({"protocol_version": version, "created_at": now_utc(), "parent_version": parent,
                    "test_opened": False, "plan_hash": None})
    dump_json(entries, path)


def write_plan_lock(out_dir: Path, version: str, runs: Sequence[Mapping], *,
                    registry_path: Path | None = None) -> str:
    """Write plan.lock.json and return plan_hash; once test is opened the plan cannot change."""
    out_dir = Path(out_dir)
    ordered = sorted((dict(r) for r in runs), key=lambda r: r["run_id"])
    plan_hash = sha256_text(canonical_json({"protocol_version": version, "runs": ordered}))
    lock_path = out_dir / PLAN_LOCK
    if lock_path.exists():
        old = load_json(lock_path)
        if old["protocol_version"] != version:
            raise ProtocolViolationError(f"{lock_path} holds the plan of protocol version {old['protocol_version']!r}; "
                                         "use one plan directory per protocol version")
        if old["plan_hash"] == plan_hash:
            return plan_hash
        if (out_dir / TEST_OPENED).exists():
            raise ProtocolViolationError(f"{lock_path}: test is open for this plan; it cannot be changed (code 5)")
    entries = entry = None
    if registry_path is not None:
        entries = load_registry(registry_path)
        entry = _entry(entries, version, Path(registry_path))
        if entry["test_opened"] and entry["plan_hash"] != plan_hash:   # wherever the new lock would be written
            raise ProtocolViolationError(f"protocol version {version!r} has an open test; its plan cannot change "
                                         "(code 5) — register a new version")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json({"protocol_version": version, "plan_hash": plan_hash, "created_at": now_utc(),
               "config_hashes": sorted({r["config_hash"] for r in ordered}),
               "run_ids": [r["run_id"] for r in ordered]}, lock_path)
    if entry is not None and entry["plan_hash"] != plan_hash:
        entry["plan_hash"] = plan_hash
        dump_json(entries, registry_path)
    return plan_hash


def check_run_in_plan(out_dir: Path, config_hash: str) -> None:
    lock_path = Path(out_dir) / PLAN_LOCK
    if not lock_path.is_file():
        raise ProtocolViolationError(f"{lock_path} is missing: confirmatory runs need a frozen plan (experiments plan)")
    if config_hash not in load_json(lock_path)["config_hashes"]:
        raise ProtocolViolationError(f"config_hash {config_hash[:12]}… is not in {lock_path}")


def _opened(registry_path: Path, version: str) -> bool:
    return bool(_entry(load_registry(registry_path), version, Path(registry_path))["test_opened"])


def open_test(registry_path: Path, out_dir: Path, version: str, run_id: str) -> None:
    """Record the first access to test (test_opened.json) and mark the version; later calls keep the first marker."""
    entries = load_registry(registry_path)
    entry = _entry(entries, version, Path(registry_path))
    marker = Path(out_dir) / TEST_OPENED
    if not marker.exists():
        dump_json({"protocol_version": version, "run_id": run_id, "opened_at": now_utc(),
                   "plan_hash": entry["plan_hash"]}, marker)
    if not entry["test_opened"]:
        entry["test_opened"] = True
        dump_json(entries, registry_path)


def is_post_test(registry_path: Path, version: str) -> bool:
    """True when any ancestor version has an open test: results are a separate study (§7.5)."""
    entries = load_registry(registry_path)
    parent = _entry(entries, version, Path(registry_path))["parent_version"]
    while parent is not None:
        e = _entry(entries, parent, Path(registry_path))
        if e["test_opened"]:
            return True
        parent = e["parent_version"]
    return False


def assert_plan_mutable(registry_path: Path, version: str) -> None:
    if _opened(registry_path, version):
        raise ProtocolViolationError(f"protocol version {version!r} has an open test; its plan cannot change (code 5)")
