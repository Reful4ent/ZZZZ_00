import json
from pathlib import Path

import pytest

from expcore.errors import DataValidationError, ProtocolViolationError, TestAccessError
from expcore.io import load_json
from expcore.loaders import SplitGuard
from expcore.protocol import (assert_plan_mutable, check_run_in_plan, is_post_test, load_registry, open_test,
                              register_version, write_plan_lock)

pytestmark = pytest.mark.unit   # T-REP-04

REPO = Path(__file__).resolve().parents[2]
RUNS = [{"module": "photometric", "seed": s, "config_hash": "c" * 64, "run_id": f"photometric-confirmatory-cccc-s{s}"}
        for s in (11, 23)]


@pytest.fixture
def reg(tmp_path):
    path = tmp_path / "protocol_registry.json"
    register_version(path, "1", None)
    return path


def test_shipped_registry_is_valid():
    entries = load_registry(REPO / "configs" / "protocol_registry.json")
    assert [e["protocol_version"] for e in entries] == ["1"] and entries[0]["test_opened"] is False


def test_new_version_required_for_grid_change(reg):
    register_version(reg, "2", "1")
    with pytest.raises(ProtocolViolationError):
        register_version(reg, "2", "1")
    with pytest.raises(ProtocolViolationError):
        register_version(reg, "3", "unknown-parent")
    assert [e["protocol_version"] for e in load_registry(reg)] == ["1", "2"]


def test_confirmatory_run_must_be_in_plan(reg, tmp_path):
    out = tmp_path / "plan"
    with pytest.raises(ProtocolViolationError, match="plan.lock.json"):
        check_run_in_plan(out, "c" * 64)
    plan_hash = write_plan_lock(out, "1", RUNS, registry_path=reg)
    assert load_json(out / "plan.lock.json")["plan_hash"] == plan_hash == load_registry(reg)[0]["plan_hash"]
    check_run_in_plan(out, "c" * 64)
    with pytest.raises(ProtocolViolationError):
        check_run_in_plan(out, "d" * 64)
    assert write_plan_lock(out, "1", list(reversed(RUNS))) == plan_hash   # order-independent, idempotent


def test_first_test_access_creates_marker(reg, tmp_path):
    out = tmp_path / "plan"
    write_plan_lock(out, "1", RUNS)
    open_test(reg, out, "1", RUNS[0]["run_id"])
    marker = load_json(out / "test_opened.json")
    assert marker["run_id"] == RUNS[0]["run_id"] and marker["protocol_version"] == "1"
    assert load_registry(reg)[0]["test_opened"] is True
    open_test(reg, out, "1", RUNS[1]["run_id"])   # later accesses keep the first marker
    assert load_json(out / "test_opened.json")["run_id"] == RUNS[0]["run_id"]


def test_plan_immutable_after_test_open(reg, tmp_path):
    out = tmp_path / "plan"
    assert_plan_mutable(reg, "1")
    write_plan_lock(out, "1", RUNS)
    open_test(reg, out, "1", RUNS[0]["run_id"])
    with pytest.raises(ProtocolViolationError):
        assert_plan_mutable(reg, "1")
    with pytest.raises(ProtocolViolationError):   # the lock itself cannot be changed either
        write_plan_lock(out, "1", RUNS[:1], registry_path=reg)


def test_child_version_marked_post_test(reg, tmp_path):
    register_version(reg, "2", "1")
    assert is_post_test(reg, "2") is False
    out = tmp_path / "plan"
    write_plan_lock(out, "1", RUNS)
    open_test(reg, out, "1", RUNS[0]["run_id"])
    register_version(reg, "3", "2")
    assert is_post_test(reg, "2") is True and is_post_test(reg, "3") is True and is_post_test(reg, "1") is False


def test_guard_denies_test_before_open():
    guard = SplitGuard("confirmatory", fixture=False)
    with pytest.raises(TestAccessError):
        guard.check("test")


def test_corrupt_registry_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"protocol_version": "1"}]), encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_registry(p)
