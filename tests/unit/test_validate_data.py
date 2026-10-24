import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from expcore.config import load_config
from expcore.io import load_json, read_parquet, write_parquet
from expcore.manifests import FROZEN, manifest_hash
from expcore.schemas import IMAGES_COLUMNS, IMAGES_NULLABLE
from experiments.cli import main
from tests.cli_helpers import make_world

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def _report(world):
    return load_json(world.out_root / "smoke" / "validate-data" / "v1" / "validation_report.json")


def _vdir(world):
    return Path(world.cfg_dict["paths"]["manifest_dir"]) / "v1"


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch, build=False)
    assert main(w.argv("build-manifests")) == 0
    return w


def test_build_then_validate_ok(world):
    assert (_vdir(world) / FROZEN).is_file()
    assert main(world.argv("validate-data")) == 0
    rep = _report(world)
    assert rep["ok"] is True and rep["issues"] == [] and rep["version"] == "v1"
    assert main(world.argv("build-manifests")) == 5   # an existing version is never rebuilt


def test_validate_without_manifest_exit_3(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch, build=False)
    assert main(w.argv("validate-data")) == 3


def test_validate_detects_deleted_image(world):
    images = read_parquet(_vdir(world) / "images.parquet")
    victim = images.relative_path.iloc[0]
    (Path(world.cfg_dict["paths"]["data_root"]) / victim).unlink()
    assert main(world.argv("validate-data")) == 3
    rep = _report(world)
    missing = next(i for i in rep["issues"] if i["check"] == "files_present")
    assert victim in missing["items"]


def test_validate_detects_modified_image_sha(world):
    images = read_parquet(_vdir(world) / "images.parquet")
    victim = Path(world.cfg_dict["paths"]["data_root"]) / images.relative_path.iloc[3]
    victim.write_bytes(victim.read_bytes() + b"\0")
    assert main(world.argv("validate-data")) == 3
    assert any(i["check"] == "files_sha256" for i in _report(world)["issues"])


def test_validate_detects_split_drift(world):
    vdir = _vdir(world)
    images = read_parquet(vdir / "images.parquet")
    sop = images.index[(images.dataset_id == "sop") & (images.split == "train")]
    group = images.loc[sop[0], "source_group_id"]
    images.loc[(images.dataset_id == "sop") & (images.source_group_id == group), "split"] = "validation"
    write_parquet(images, vdir / "images.parquet", schema=IMAGES_COLUMNS, nullable=IMAGES_NULLABLE)
    frozen = load_json(vdir / FROZEN)
    pairs = {p.name: read_parquet(p / "pairs.parquet") for p in vdir.iterdir() if (p / "pairs.parquet").is_file()}
    frozen["manifest_hash"] = manifest_hash(images, load_json(vdir / "datasets.json"), pairs)   # fake the freeze
    (vdir / FROZEN).write_text(__import__("json").dumps(frozen), encoding="utf-8")
    assert main(world.argv("validate-data")) == 3
    issues = {i["check"]: i for i in _report(world)["issues"]}
    assert "manifest_hash" not in issues and "splits_recomputed" in issues


def test_validate_blocks_on_cross_split_duplicates(world):   # T-DAT-06 end-to-end
    images = read_parquet(_vdir(world) / "images.parquet")
    kadid = images[images.dataset_id == "kadid10k"]
    a = kadid[kadid.split == "validation"].relative_path.iloc[0]
    b = kadid[kadid.split == "test"].relative_path.iloc[0]
    root = Path(world.cfg_dict["paths"]["data_root"])
    shutil.copy(root / a, root / b)
    assert main(world.argv("validate-data")) == 3
    issues = {i["check"]: i for i in _report(world)["issues"]}
    assert "cross_split_duplicates" in issues and "files_sha256" in issues


def test_build_blocking_duplicate_exit_3(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch, build=False)
    root = Path(w.cfg_dict["paths"]["data_root"]) / "sop"
    first_train = (root / "Ebay_train.txt").read_text(encoding="utf-8").splitlines()[1].split()[3]
    first_test = (root / "Ebay_test.txt").read_text(encoding="utf-8").splitlines()[1].split()[3]
    shutil.copy(root / first_train, root / first_test)
    assert main(w.argv("build-manifests")) == 3   # LeakageError: not frozen
    assert not (_vdir(w) / FROZEN).exists()


def test_build_with_overrides_file(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch, build=False)
    assert main(w.argv("build-manifests")) == 0
    images = read_parquet(_vdir(w) / "images.parquet")
    sha = images[images.dataset_id == "sop"].sha256.iloc[0]   # an HPatches view would make the delivery a subset
    for ds in w.cfg_dict["datasets"].values():
        ds["manifest_version"] = "v2"
    w.write("cfg2.yaml")
    ov = tmp_path / "overrides.yaml"
    ov.write_text(yaml.safe_dump({"exclude_image": [{"sha256": sha, "reason": "test"}]}), encoding="utf-8")
    assert main([*w.argv("build-manifests"), "--overrides", str(ov)]) == 0
    assert load_json(Path(w.cfg_dict["paths"]["manifest_dir"]) / "v2" / "exclusions.json")[0]["reason"] == "test"
    assert main(w.argv("validate-data")) == 0   # exclusions are re-applied before recomputing splits


def test_provenance_fields_present(world):   # T-DAT-08
    for rec in load_json(_vdir(world) / "datasets.json"):
        for field in ("source_url", "download_date", "archive_name", "archive_sha256", "license", "license_file",
                      "discovered_counts"):
            assert field in rec


def test_repo_contains_no_dataset_images():   # T-DAT-08
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout
    assert [f for f in tracked.splitlines() if f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm"))] == []


def test_help_exits_zero():
    import sys
    for cmd in ("build-manifests", "validate-data"):
        assert subprocess.run([sys.executable, "-m", "experiments", cmd, "--help"], capture_output=True,
                              cwd=REPO).returncode == 0
