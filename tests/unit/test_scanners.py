import dataclasses
import shutil
from pathlib import Path

import pytest

from expcore.errors import ConfigError, CountMismatchError, DataValidationError, LeakageError
from expcore.scanners import RAW_COLUMNS, check_counts, hpatches_completeness, read_image_meta, scan, to_relative_path
from tests.fixtures_builder import make_hpatches, make_kadid, make_mvtec, make_sop

pytestmark = pytest.mark.unit


def test_scan_kadid_counts(kadid_root, fixture_cfg):
    raw = scan(fixture_cfg.datasets["kadid10k"], kadid_root)
    assert list(raw.columns) == list(RAW_COLUMNS)
    assert (raw.role == "original").sum() == 6 and (raw.role == "derivative").sum() == 24
    der = raw[raw.role == "derivative"].iloc[0]
    assert der.source_group_id == der.natural_key[:3] and der.category_id == "01"
    assert raw.relative_path.str.startswith("kadid10k/images/").all()
    assert (raw[["width", "height", "channels"]].to_numpy() == [16, 16, 3]).all()
    assert check_counts(raw, fixture_cfg.datasets["kadid10k"])["discovered"] == {"originals": 6, "derivatives": 24}


def test_kadid_count_mismatch(kadid_root, fixture_cfg):
    wrong = dataclasses.replace(fixture_cfg.datasets["kadid10k"], expected={"originals": 81, "derivatives": 10125})
    with pytest.raises(CountMismatchError, match="originals"):
        check_counts(scan(wrong, kadid_root), wrong)


def test_unknown_expected_key(kadid_root, fixture_cfg):
    bad = dataclasses.replace(fixture_cfg.datasets["kadid10k"], expected={"originalz": 6})
    with pytest.raises(ConfigError, match="originalz"):
        check_counts(scan(bad, kadid_root), bad)


def test_kadid_unexpected_file(tmp_path, fixture_cfg):
    make_kadid(tmp_path / "kadid10k", 2, 1, 1)
    (tmp_path / "kadid10k" / "images" / "readme.png").write_bytes(b"x")
    with pytest.raises(DataValidationError, match="readme.png"):
        scan(fixture_cfg.datasets["kadid10k"], tmp_path)


def test_mvtec_scan_excludes_unused_categories(tmp_path, fixture_cfg):
    make_mvtec(tmp_path / "mvtec_ad", categories=("carpet", "leather", "tile", "wood", "bottle"))
    raw = scan(fixture_cfg.datasets["mvtec_ad"], tmp_path)
    assert set(raw.category_id) == {"carpet", "leather", "tile", "wood"}
    assert set(raw.role) == {"train_good", "test"}
    assert set(raw.defect_type) == {"good", "crack"}
    # USER DECISION 2a: groups carry the subdirectory, so train/good/000 and test/crack/000 differ
    groups = set(raw[raw.category_id == "carpet"].source_group_id)
    assert {"carpet/train/good/000", "carpet/test/good/000", "carpet/test/crack/000"} <= groups
    assert raw.source_group_id.is_unique
    assert check_counts(raw, fixture_cfg.datasets["mvtec_ad"])["discovered"] == {
        "train_good": 32, "test_good": 8, "test_defect": 8}


def test_mvtec_defect_without_mask(tmp_path, fixture_cfg):
    make_mvtec(tmp_path / "mvtec_ad")
    (tmp_path / "mvtec_ad" / "tile" / "ground_truth" / "crack" / "001_mask.png").unlink()
    with pytest.raises(DataValidationError, match="001_mask.png"):
        scan(fixture_cfg.datasets["mvtec_ad"], tmp_path)


def test_hpatches_scan_and_full(hpatches_root, fixture_cfg):
    cfg = fixture_cfg.datasets["hpatches"]
    raw = scan(cfg, hpatches_root)
    assert len(raw) == 60 and set(raw.role) == {"sequence_image"} and set(raw.category_id) == {"i", "v"}
    rep = hpatches_completeness(raw, cfg, hpatches_root)
    assert rep["status"] == "full" and rep["incomplete"] == [] and len(rep["sequences"]) == 10


def test_hpatches_incomplete_is_subset(tmp_path, fixture_cfg):   # T-DAT-07
    make_hpatches(tmp_path / "hpatches", complete=False)
    cfg = dataclasses.replace(fixture_cfg.datasets["hpatches"], expected={"sequences": 10, "subset_name": "mini"})
    rep = hpatches_completeness(scan(cfg, tmp_path), cfg, tmp_path)
    assert rep["status"] == "subset" and rep["incomplete"] == ["v_seq00"] and rep["subset_name"] == "mini"
    check_counts(scan(cfg, tmp_path), cfg)   # HPatches: completeness, never CountMismatchError


def test_hpatches_without_expected_is_unverified(hpatches_root, fixture_cfg):
    cfg = dataclasses.replace(fixture_cfg.datasets["hpatches"], expected={})
    assert hpatches_completeness(scan(cfg, hpatches_root), cfg, hpatches_root)["status"] == "unverified"


def test_corrupt_image_names_file(tmp_path):      # Review Focus 5
    (tmp_path / "I01.png").write_bytes(b"not an image")
    with pytest.raises(DataValidationError, match="I01.png"):
        read_image_meta(tmp_path / "I01.png")


def test_unexpected_channels(tmp_path):           # Review Focus 5
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((4, 4, 4), np.uint8))
    with pytest.raises(DataValidationError, match="channels"):
        read_image_meta(tmp_path / "a.png")


def test_paths_with_spaces_unicode_normalized(tmp_path, fixture_cfg):   # Review Focus 4
    assert to_relative_path(Path("root") / "папка с пробелом" / "x.png", Path("root")) == "папка с пробелом/x.png"
    assert to_relative_path("root\\папка с пробелом\\x.png", "root") == "папка с пробелом/x.png"
    data_root = tmp_path / "данные с пробелом"
    make_mvtec(data_root / "mvtec_ad")
    good = data_root / "mvtec_ad" / "carpet" / "train" / "good"
    shutil.copy(good / "000.png", good / "копия 1.png")
    raw = scan(fixture_cfg.datasets["mvtec_ad"], data_root)
    row = raw[raw.relative_path == "mvtec_ad/carpet/train/good/копия 1.png"]
    assert len(row) == 1 and row.iloc[0].width == 16


def test_sop_scan(sop_root, fixture_cfg):
    raw = scan(fixture_cfg.datasets["sop"], sop_root)
    assert set(raw.role) == {"train", "test"} and raw.class_id.map(type).eq(str).all()
    assert (raw.source_group_id == raw.class_id).all()
    assert check_counts(raw, fixture_cfg.datasets["sop"])["discovered"] == {
        "products": 14, "train_images": 30, "test_images": 12}


def test_sop_class_disjoint(tmp_path, fixture_cfg):
    make_sop(tmp_path / "sop")
    test_txt = tmp_path / "sop" / "Ebay_test.txt"
    lines = test_txt.read_text(encoding="utf-8").splitlines()
    first = lines[1].split()
    lines[1] = " ".join([first[0], "1", first[2], first[3]])   # class 1 is a train class
    test_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LeakageError):
        scan(fixture_cfg.datasets["sop"], tmp_path)


def test_hpatches_empty_sequence_dir_listed(tmp_path, fixture_cfg):   # review P3-I2: ДАН-03 lists all sequences
    make_hpatches(tmp_path / "hpatches")
    (tmp_path / "hpatches" / "v_empty").mkdir()
    (tmp_path / "hpatches" / "v_honly").mkdir()
    (tmp_path / "hpatches" / "v_honly" / "H_1_2").write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    cfg = dataclasses.replace(fixture_cfg.datasets["hpatches"], expected={})
    rep = hpatches_completeness(scan(cfg, tmp_path), cfg, tmp_path)
    assert {"v_empty", "v_honly"} <= set(rep["sequences"]) and rep["incomplete"] == ["v_empty", "v_honly"]
    assert rep["status"] == "subset"
