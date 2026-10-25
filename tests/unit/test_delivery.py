import importlib
import importlib.util
import re
from pathlib import Path

import pytest

from expcore.config import load_config
from expcore.reporting import DISCLAIMER

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ALL_COMMANDS = [f"python -m experiments {c}" for c in
                ("validate-data", "build-manifests", "plan", "run", "aggregate", "report")] + [
    "python -m photometric run", "python -m geometric run", "python -m metric train", "python -m metric embed",
    "python -m metric search"]


def _gen_schemas():
    spec = importlib.util.spec_from_file_location("gen_schemas", REPO / "scripts" / "gen_schemas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_schemas_match_generated(tmp_path):
    generated = {p.name: p.read_bytes() for p in _gen_schemas().generate(tmp_path)}
    committed = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in (REPO / "schemas").glob("*.schema.json")}
    assert set(generated) == {f"{n}.schema.json" for n in
                              ("datasets", "images", "pairs", "predictions", "run", "thresholds", "metrics")}
    assert generated == committed


def test_readme_has_disclaimer_and_all_commands():   # T-REP-01, §11.3 п.5
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert DISCLAIMER in text and all(c in text for c in ALL_COMMANDS)


def test_decisions_has_all_D_entries():
    ids = set(re.findall(r"^\| (D-\d\d) \|", (REPO / "DECISIONS.md").read_text(encoding="utf-8"), re.M))
    assert {f"D-{i:02d}" for i in range(1, 21)} <= ids


@pytest.mark.parametrize("name", ["photometric", "geometric", "metric", "experiments"])
def test_smoke_configs_load(name):
    importlib.import_module(f"{name}.cli")
    cfg = load_config(REPO / "configs" / "smoke" / f"{name}.yaml", module=name)
    assert cfg.profile == "smoke" and all(ds.fixture for ds in cfg.datasets.values())


def test_smoke_configs_use_a_registered_protocol_version():
    from expcore.protocol import load_registry
    versions = {e["protocol_version"] for e in load_registry(REPO / "configs" / "protocol_registry.json")}
    importlib.import_module("experiments.cli")
    assert load_config(REPO / "configs" / "smoke" / "experiments.yaml", module="experiments").protocol_version in versions


def test_full_template_loads():
    from tests.conftest import TEST_MODULE
    cfg = load_config(REPO / "configs" / "templates" / "full.yaml", module=TEST_MODULE)
    assert cfg.profile == "pilot"


def test_experiments_cli_imports_program_packages():   # TZ03 request 1: build-manifests sees their pair builders
    import subprocess
    import sys
    code = ("import sys, experiments.cli; "
            "assert {'photometric', 'geometric', 'metric'} <= set(sys.modules); "
            "assert 'tensorflow' not in sys.modules")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=REPO).returncode == 0
