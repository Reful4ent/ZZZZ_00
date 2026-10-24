import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from expcore.config import load_config, register_params_schema
from tests.fixtures_builder import make_all

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs" / "templates" / "full.yaml"
TEST_MODULE = "test"   # params schema used by fixture configs


@dataclass(frozen=True)
class TestParams:
    __test__ = False


register_params_schema(TEST_MODULE, TestParams)


def smoke_cfg_dict(counts: dict[str, dict[str, int]], data_root: Path, out_root: Path) -> dict:
    """The full template turned into a smoke config over fixture datasets with the given counts."""
    d = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    d["profile"] = "smoke"
    d["seeds"] = {"seed": 11, "seeds": [11]}
    for dataset_id, ds in d["datasets"].items():
        ds["fixture"] = True
        ds["expected"] = copy.deepcopy(counts[dataset_id])
    d["datasets"]["kadid10k"]["n_validation"] = 2
    d["paths"] = {"data_root": str(data_root), "manifest_dir": str(out_root / "manifests"), "out_root": str(out_root)}
    return d


def write_cfg(d: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory):
    """(data_root, counts) — all four fixture datasets under data_root/<dataset_id>/. Read-only."""
    root = tmp_path_factory.mktemp("data")
    return root, make_all(root)


@pytest.fixture(scope="session")
def data_root(fixture_data):
    return fixture_data[0]


# Scanners take the shared data_root and read data_root/<dataset_id>/.
@pytest.fixture(scope="session")
def kadid_root(data_root):
    return data_root


@pytest.fixture(scope="session")
def hpatches_root(data_root):
    return data_root


@pytest.fixture(scope="session")
def mvtec_root(data_root):
    return data_root


@pytest.fixture(scope="session")
def sop_root(data_root):
    return data_root


@pytest.fixture
def fixture_cfg_dict(fixture_data, tmp_path):
    root, counts = fixture_data
    return smoke_cfg_dict(counts, root, tmp_path / "out")


@pytest.fixture
def fixture_cfg_path(fixture_cfg_dict, tmp_path):
    return write_cfg(fixture_cfg_dict, tmp_path / "smoke.yaml")


@pytest.fixture
def fixture_cfg(fixture_cfg_path):
    return load_config(fixture_cfg_path, module=TEST_MODULE)
