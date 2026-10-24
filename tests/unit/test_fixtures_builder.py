import pytest

from tests.fixtures_builder import make_hpatches, make_kadid, make_mvtec, make_sop, sha_tree

pytestmark = pytest.mark.unit


def test_fixtures_deterministic_and_complete(tmp_path):
    for make in (make_kadid, make_hpatches, make_mvtec, make_sop):
        a, b = tmp_path / make.__name__ / "a", tmp_path / make.__name__ / "b"
        assert make(a) == make(b)
        assert sha_tree(a) == sha_tree(b) and sha_tree(a)

    counts = make_kadid(tmp_path / "c", 6, 2, 2)
    assert counts == {"originals": 6, "derivatives": 6 * 2 * 2}
    assert len(list((tmp_path / "c" / "images").glob("*.png"))) == 6 + 24

    counts = make_hpatches(tmp_path / "h", n_i=2, n_v=3)
    assert counts == {"sequences": 5}
    for seq in (tmp_path / "h").iterdir():
        assert sorted(p.name for p in seq.iterdir()) == ["1.ppm", "2.ppm", "3.ppm", "4.ppm", "5.ppm", "6.ppm",
                                                        "H_1_2", "H_1_3", "H_1_4", "H_1_5", "H_1_6"]
    make_hpatches(tmp_path / "hi", complete=False)
    assert sum(len(list(s.iterdir())) < 11 for s in (tmp_path / "hi").iterdir()) == 1

    counts = make_mvtec(tmp_path / "m", categories=("carpet", "tile"))
    assert counts == {"train_good": 16, "test_good": 4, "test_defect": 4}
    defects = sorted((tmp_path / "m" / "carpet" / "test").glob("*/*.png"))
    masks = {p.name for p in (tmp_path / "m" / "carpet" / "ground_truth").glob("*/*_mask.png")}
    assert {f"{p.stem}_mask.png" for p in defects if p.parent.name != "good"} == masks and masks

    counts = make_sop(tmp_path / "s", n_train_classes=10, n_test_classes=4, per_class=3)
    assert counts == {"products": 14, "train_images": 30, "test_images": 12}
    lines = (tmp_path / "s" / "Ebay_train.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "image_id class_id super_class_id path" and len(lines) - 1 == 30
    assert all((tmp_path / "s" / ln.split()[3]).is_file() for ln in lines[1:])


def test_all_images_distinct(tmp_path):   # identical bytes across splits would be a blocking duplicate
    for make in (make_kadid, make_hpatches, make_mvtec, make_sop):
        make(tmp_path / make.__name__)
    hashes = [h for p, h in sha_tree(tmp_path).items() if not p.endswith((".txt", "_mask.png")) and "/H_1_" not in p]   # masks/H are not images
    assert len(hashes) == len(set(hashes))


def test_fixture_cfg_is_valid_smoke_config(fixture_cfg, data_root):
    assert fixture_cfg.profile == "smoke" and all(ds.fixture for ds in fixture_cfg.datasets.values())
    assert fixture_cfg.datasets["kadid10k"].expected == {"originals": 6, "derivatives": 24}
    assert fixture_cfg.datasets["kadid10k"].n_validation == 2
    assert all((data_root / d).is_dir() for d in ("kadid10k", "hpatches", "mvtec_ad", "sop"))
