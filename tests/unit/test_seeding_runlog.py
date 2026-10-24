import json
import subprocess
import sys

import numpy as np
import pytest

from expcore.errors import ProtocolViolationError, ResultTamperedError
from expcore.io import dump_json, load_json
from expcore.runlog import JsonlLogger, RunRecord, collect_environment, git_state, make_run_id
from expcore.schemas import validate_run_json
from expcore.seeding import STREAM_INDEX, capture_rng_state, init_seeds, stream
from expcore.timing import peak_memory_mb, timed

pytestmark = pytest.mark.unit

H = "a" * 64


def test_streams_independent_of_call_order():
    a = stream(11, "sampler", 3).random()
    stream(11, "sampler", 0).random()
    assert stream(11, "sampler", 3).random() == a


def test_streams_differ_by_name_and_unit():
    values = {(n, u): stream(11, n, u).random() for n in STREAM_INDEX for u in (0, 1)}
    assert len(set(values.values())) == len(values)
    assert stream(11, "sampler", 0).random() != stream(23, "sampler", 0).random()
    with pytest.raises(ValueError):
        stream(11, "unknown", 0)


def test_init_seeds_reproducible():
    import random
    init_seeds(11); x, y = np.random.rand(), random.random()
    init_seeds(11)
    assert np.random.rand() == x and random.random() == y


def test_rng_state_json_serializable(tmp_path):
    init_seeds(11)
    state = capture_rng_state()
    dump_json(state, tmp_path / "rng_state.json")
    assert load_json(tmp_path / "rng_state.json")["opencv"] == {"seed": 11, "state": None,
                                                                "note": "opencv_state_not_readable"}


def test_run_id_format():
    assert make_run_id("geometric", "pilot", "a" * 64, 23) == "geometric-pilot-aaaaaaaaaaaa-s23"


def _record(run_dir, **kw):
    fields = dict(run_id=make_run_id("photometric", "pilot", H, 11), config_hash=H, dataset_manifest_hash=H, seed=11,
                  profile="pilot", module="photometric", command="run", protocol_version="1", oracle_mode=False,
                  post_test_study=False, git_commit="abc123", dirty_diff_hash=None,
                  environment=collect_environment(deterministic_ops=None), library_primitives=[])
    fields.update(kw)
    if "run_id" not in kw:
        fields["run_id"] = make_run_id(fields["module"], fields["profile"], fields["config_hash"], fields["seed"])
    return RunRecord(run_dir, **fields)


def test_run_json_valid_and_status_transitions(tmp_path):
    rec = _record(tmp_path)
    rec.start()
    run = load_json(tmp_path / "run.json")
    validate_run_json(run)
    assert run["status"] == "running" and run["finished_at"] is None and len(run["attempts"]) == 1
    assert run["environment"]["path"] == "environment.json" and (tmp_path / "environment.json").is_file()
    rec.finish("completed")
    run = load_json(tmp_path / "run.json")
    validate_run_json(run)
    assert run["status"] == "completed" and run["finished_at"] and run["attempts"][0]["status"] == "completed"


def test_failed_then_resume_appends_attempt(tmp_path):
    rec = _record(tmp_path)
    rec.start()
    rec.finish("failed", {"type": "TechnicalError", "message": "boom", "unit_id": 2})
    first = load_json(tmp_path / "run.json")
    validate_run_json(first)
    assert first["failure"]["unit_id"] == 2
    again = _record(tmp_path)
    again.start()
    run = load_json(tmp_path / "run.json")
    assert run["status"] == "running" and run["failure"] is None and len(run["attempts"]) == 2
    assert run["started_at"] == first["started_at"] and run["attempts"][0]["status"] == "failed"
    again.finish("completed")
    assert load_json(tmp_path / "run.json")["status"] == "completed"
    with pytest.raises(ProtocolViolationError):   # a completed run is not started again
        _record(tmp_path).start()


def test_resume_with_other_identity_refused(tmp_path):
    rec = _record(tmp_path)
    rec.start()
    rec.finish("failed", {"type": "TechnicalError", "message": "x", "unit_id": None})
    with pytest.raises(ResultTamperedError):
        _record(tmp_path, dataset_manifest_hash="b" * 64).start()


def test_confirmatory_requires_git_commit(tmp_path):
    with pytest.raises(ProtocolViolationError):
        _record(tmp_path, profile="confirmatory", git_commit=None)
    _record(tmp_path, profile="smoke", git_commit=None).start()


def test_jsonl_logger(tmp_path):
    log = JsonlLogger(tmp_path / "run.log.jsonl", "r1")
    log.log("INFO", "unit_done", unit_id=0, n_rows=3)
    log.log("ERROR", "failed", message="x")
    lines = [json.loads(s) for s in (tmp_path / "run.log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [ln["event"] for ln in lines] == ["unit_done", "failed"]
    assert set(lines[0]) == {"ts_utc", "level", "run_id", "event", "payload"} and lines[0]["payload"]["n_rows"] == 3
    with pytest.raises(ValueError):
        log.log("INFO", "bad", value=float("nan"))


def test_environment_fields():
    env = collect_environment(deterministic_ops=False, extra={"library_primitives": [{"name": "sift"}]})
    for key in ("python", "os", "cpu", "cpu_count", "ram_total_mb", "gpu", "libraries", "env_vars",
                "nondeterminism_notes", "library_primitives", "deterministic_ops"):
        assert key in env
    assert set(env["env_vars"]) == {"PYTHONHASHSEED", "TF_DETERMINISTIC_OPS", "CUDA_VISIBLE_DEVICES"}
    assert "numpy" in env["libraries"] and env["library_primitives"] == [{"name": "sift"}]
    assert any("deterministic" in n for n in env["nondeterminism_notes"])
    json.dumps(env, allow_nan=False)


def test_git_state(tmp_path):
    assert git_state(tmp_path) == (None, None)   # not a repository
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q"); git("config", "user.email", "t@example.org"); git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1", encoding="utf-8")
    git("add", "a.txt"); git("commit", "-q", "-m", "x")
    commit, dirty = git_state(tmp_path)
    assert len(commit) == 40 and dirty is None
    (tmp_path / "a.txt").write_text("2", encoding="utf-8")
    _, d1 = git_state(tmp_path)
    (tmp_path / "new.txt").write_text("u", encoding="utf-8")
    _, d2 = git_state(tmp_path)
    assert len(d1) == 64 and d1 != d2


def test_timed():
    result, ms = timed(sum, [1, 2, 3])
    assert result == 6 and ms >= 0.0


def test_peak_memory_null_reason_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert peak_memory_mb() == (None, "psutil_unavailable")
