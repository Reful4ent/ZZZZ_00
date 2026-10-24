import dataclasses
import shutil

import pandas as pd
import pytest

from expcore import registry
from expcore.config import load_config
from expcore.errors import ConfigError, DataValidationError, LeakageError, ProtocolViolationError
from expcore.io import load_json, read_parquet, write_parquet
from expcore.manifests import assign_splits, build_manifests, load_manifest, manifest_hash, write_pairs
from expcore.scanners import scan
from expcore.schemas import EVAL_ONLY_IMAGE_COLUMNS, IMAGES_COLUMNS, IMAGES_NULLABLE
from tests.conftest import TEST_MODULE, smoke_cfg_dict, write_cfg
from tests.fixtures_builder import make_all

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def built(tmp_path_factory, fixture_data):
    """One smoke build over the shared fixture data (read-only for tests)."""
    data_root, counts = fixture_data
    out = tmp_path_factory.mktemp("built")
    cfg = load_config(write_cfg(smoke_cfg_dict(counts, data_root, out), out / "cfg.yaml"), module=TEST_MODULE)
    info = build_manifests(cfg, data_root, out / "manifests")
    return cfg, info, load_manifest(out / "manifests", info.version)


@pytest.fixture
def own_data(tmp_path):
    """A private, mutable copy of the fixture datasets with its own config."""
    data_root = tmp_path / "data"
    counts = make_all(data_root)
    d = smoke_cfg_dict(counts, data_root, tmp_path / "out")
    return data_root, d


def _cfg(d, tmp_path, name="cfg.yaml"):
    return load_config(write_cfg(d, tmp_path / name), module=TEST_MODULE)


def test_build_all_four_datasets_smoke(built):
    _, info, m = built
    assert set(m.images.dataset_id) == {"kadid10k", "hpatches", "mvtec_ad", "sop"} and m.images.image_id.is_unique
    assert list(m.images.columns) == list(IMAGES_COLUMNS) and info.manifest_hash == m.manifest_hash
    assert sorted(f.name for f in info.dir.iterdir()) == sorted(
        ["datasets.json", "images.parquet", "manifest.frozen.json", "completeness_report.json",
         "duplicates_report.json", "exclusions.json"])
    assert {d["dataset_id"] for d in m.datasets} == {"kadid10k", "hpatches", "mvtec_ad", "sop"}


def test_split_shares(built):
    _, _, m = built
    sizes = m.images.groupby(["dataset_id", "split"]).size().to_dict()
    assert sizes[("kadid10k", "validation")] == 2 * 5 and sizes[("kadid10k", "test")] == 4 * 5   # n_validation=2
    assert sizes[("mvtec_ad", "dev")] == 4 * 2 and sizes[("mvtec_ad", "synthetic_test")] == 4 * 4
    assert sizes[("mvtec_ad", "real_test")] == 4 * 4
    assert sizes[("sop", "train")] == 8 * 3 and sizes[("sop", "validation")] == 2 * 3 and sizes[("sop", "test")] == 12
    assert sizes[("hpatches", "validation")] == 2 * 6


def test_kadid_derivatives_follow_original(built):
    _, _, m = built
    kadid = m.images[m.images.dataset_id == "kadid10k"]
    assert (kadid.groupby("source_group_id").split.nunique() == 1).all()


def test_image_id_opaque(built):
    _, _, m = built
    stems = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in m.images.relative_path} - {""}
    stems = {s for s in stems if not s.isdigit()}   # digits alone also occur in the ordinal
    assert m.images.image_id.str.fullmatch(r"(kadid10k|hpatches|mvtec_ad|sop):\d{6}").all()
    assert not any(s in i for i in m.images.image_id for s in ("good", "defect", "crack", *stems))
    sop = m.images[m.images.dataset_id == "sop"]
    assert list(sop.sort_values("image_id").relative_path) == sorted(sop.relative_path)   # D-05


def test_frozen_marker_written_last(built):
    _, info, _ = built
    frozen = (info.dir / "manifest.frozen.json").stat().st_mtime_ns
    assert all(p.stat().st_mtime_ns <= frozen for p in info.dir.rglob("*") if p.is_file())
    marker = load_json(info.dir / "manifest.frozen.json")
    assert marker["version"] == info.version and marker["manifest_hash"] == info.manifest_hash


def test_provenance_and_counts(built):
    _, _, m = built
    hp = next(d for d in m.datasets if d["dataset_id"] == "hpatches")
    assert hp["completeness"] == "full" and hp["archive_verified"] is False and hp["archive_sha256"] is None
    kadid = next(d for d in m.datasets if d["dataset_id"] == "kadid10k")
    c = kadid["discovered_counts"]
    assert c["images"] == 30 and c["groups"] == 6 and c["splits"] == {"test": 20, "validation": 10}
    assert c["check_counts"]["matches"] is True and c["excluded"] == 0


def test_rebuild_same_version_refused(built, tmp_path):
    cfg, info, _ = built
    with pytest.raises(ProtocolViolationError):
        build_manifests(cfg, cfg.paths.data_root, info.dir.parent)


def test_tampered_images_detected(built, tmp_path):
    _, info, _ = built
    copy = tmp_path / "m"
    shutil.copytree(info.dir, copy / info.version)
    images = read_parquet(copy / info.version / "images.parquet")
    images.loc[0, "width"] = 17
    write_parquet(images, copy / info.version / "images.parquet", schema=IMAGES_COLUMNS, nullable=IMAGES_NULLABLE)
    with pytest.raises(DataValidationError, match="hash"):
        load_manifest(copy, info.version)
    load_manifest(copy, info.version, verify=False)


def test_unfrozen_manifest_refused(built, tmp_path):
    _, info, _ = built
    copy = tmp_path / "m"
    shutil.copytree(info.dir, copy / info.version)
    (copy / info.version / "manifest.frozen.json").unlink()
    with pytest.raises(DataValidationError, match="frozen"):
        load_manifest(copy, info.version)


def _kadid_pair_across_splits(images):
    kadid = images[(images.dataset_id == "kadid10k") & images.relative_path.str.contains("_")]
    val = kadid[kadid.split == "validation"].iloc[0]
    test = kadid[kadid.split == "test"].iloc[0]
    return val, test


def test_blocking_duplicate_not_frozen(built, own_data, tmp_path):
    data_root, d = own_data
    val, test = _kadid_pair_across_splits(built[2].images)
    shutil.copy(data_root / val.relative_path, data_root / test.relative_path)
    with pytest.raises(LeakageError):
        build_manifests(_cfg(d, tmp_path), data_root, tmp_path / "manifests")
    vdir = tmp_path / "manifests" / "v1"
    assert not (vdir / "manifest.frozen.json").exists()
    assert load_json(vdir / "duplicates_report.json")["blocking"] is True


def test_overrides_exclude_image_new_version(built, own_data, tmp_path):
    data_root, d = own_data
    val, test = _kadid_pair_across_splits(built[2].images)
    shutil.copy(data_root / val.relative_path, data_root / test.relative_path)
    for ds in d["datasets"].values():
        ds["manifest_version"] = "v2"
    overrides = {"exclude_image": [{"sha256": val.sha256, "reason": "identical to a test image"}]}
    info = build_manifests(_cfg(d, tmp_path), data_root, tmp_path / "manifests", overrides)
    m = load_manifest(tmp_path / "manifests", "v2")
    excl = load_json(info.dir / "exclusions.json")
    assert [e["reason"] for e in excl] == ["identical to a test image"]
    assert sorted(excl[0]["relative_paths"]) == sorted([val.relative_path, test.relative_path])
    kadid = next(x for x in m.datasets if x["dataset_id"] == "kadid10k")["discovered_counts"]
    assert kadid["images"] == 28 and kadid["excluded"] == 2
    assert val.sha256 not in set(m.images.sha256)


def test_bad_overrides(own_data, tmp_path):
    data_root, d = own_data
    cfg = _cfg(d, tmp_path)
    for bad in ({"exclude": []}, {"exclude_image": [{"sha256": "0" * 64, "reason": "x"}]},
                {"exclude_image": [{"sha256": "0" * 64}]}):
        with pytest.raises(ConfigError):
            build_manifests(cfg, data_root, tmp_path / "manifests", bad)


def test_versions_must_agree(own_data, tmp_path):
    data_root, d = own_data
    d["datasets"]["sop"]["manifest_version"] = "v9"
    with pytest.raises(ConfigError, match="manifest_version"):
        build_manifests(_cfg(d, tmp_path), data_root, tmp_path / "manifests")


def test_hpatches_subset_requires_name(tmp_path):
    from tests.fixtures_builder import make_hpatches, make_kadid, make_mvtec, make_sop
    data_root = tmp_path / "data"
    counts = {"kadid10k": make_kadid(data_root / "kadid10k"),
              "hpatches": make_hpatches(data_root / "hpatches", complete=False),
              "mvtec_ad": make_mvtec(data_root / "mvtec_ad"), "sop": make_sop(data_root / "sop")}
    d = smoke_cfg_dict(counts, data_root, tmp_path / "out")
    with pytest.raises(ConfigError, match="subset_name"):
        build_manifests(_cfg(d, tmp_path), data_root, tmp_path / "manifests")
    d["datasets"]["hpatches"]["expected"]["subset_name"] = "mini"
    m = load_manifest(tmp_path / "manifests2", build_manifests(_cfg(d, tmp_path, "b.yaml"), data_root,
                                                               tmp_path / "manifests2").version)
    hp = next(x for x in m.datasets if x["dataset_id"] == "hpatches")
    assert hp["completeness"] == "subset" and hp["subset_name"] == "mini"


def test_hash_is_stable_across_builds(built, tmp_path):
    cfg, info, _ = built
    again = build_manifests(cfg, cfg.paths.data_root, tmp_path / "elsewhere")
    assert again.manifest_hash == info.manifest_hash


def test_mvtec_real_test_and_masks_not_in_inputs(built):
    _, _, m = built
    assert "defect_type" in EVAL_ONLY_IMAGE_COLUMNS
    mvtec = m.images[m.images.dataset_id == "mvtec_ad"]
    assert set(mvtec[mvtec.split == "real_test"].defect_type) == {"good", "crack"}
    assert not mvtec.relative_path.str.contains("_mask").any()


def test_assign_splits_matches_manifest(built):
    cfg, _, m = built
    raw = assign_splits(scan(cfg.datasets["sop"], cfg.paths.data_root), cfg.datasets["sop"])
    sop = m.images[m.images.dataset_id == "sop"]
    assert dict(zip(raw.relative_path, raw.split)) == dict(zip(sop.relative_path, sop.split))


def _toy_pairs(images, cfg):
    kadid = images[images.dataset_id == "kadid10k"].sort_values("image_id")
    rows = []
    for split, g in kadid.groupby("split"):
        ids = list(g.image_id)
        rows.append(dict(pair_id=f"p1:{split}", query_id=ids[0], reference_id=ids[1], relation_id="P1_SAME_SCENE",
                         y_match=1, split=split, source_group_id=g.source_group_id.iloc[0], transform_id=None,
                         severity=None, seed=None, gt_H_path=None, gt_mask_path=None, input_mask_path=None))
    return pd.DataFrame(rows)


def test_registered_pair_builder_writes_pairs_into_hash(own_data, tmp_path, monkeypatch):
    data_root, d = own_data
    monkeypatch.setattr(registry, "_PAIR_BUILDERS", {"p1": _toy_pairs})
    info = build_manifests(_cfg(d, tmp_path), data_root, tmp_path / "manifests")
    m = load_manifest(tmp_path / "manifests", info.version)
    assert set(m.pairs) == {"p1"} and len(m.pairs["p1"]) == 2
    assert info.manifest_hash == manifest_hash(m.images, m.datasets, m.pairs)
    assert info.manifest_hash != manifest_hash(m.images, m.datasets, None)
    with pytest.raises(ProtocolViolationError):   # frozen: no more writes
        write_pairs(m.pairs["p1"], m.images, tmp_path / "manifests", info.version, "p1")
