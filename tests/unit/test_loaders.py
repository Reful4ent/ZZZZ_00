from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from expcore.errors import ConfigError, DataValidationError, ProtocolViolationError, TestAccessError
from expcore.loaders import ImageStore, PairInput, SplitGuard, load_pair_evaluation, load_pair_inputs

pytestmark = pytest.mark.unit

BASE = ["pair_id", "query_id", "reference_id", "relation_id"]


def _png(path: Path, value: int, channels: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((4, 6, channels) if channels > 1 else (4, 6), value, np.uint8)
    if channels == 3:
        img[..., 0] = 10   # distinct B and R so bgr/rgb differ
    cv2.imencode(".png", img)[1].tofile(path)


@pytest.fixture
def store_data(tmp_path):
    rows = [("kadid10k:000000", "kadid10k/images/I01.png", "validation"),
            ("kadid10k:000001", "kadid10k/images/I02.png", "test"),
            ("mvtec_ad:000000", "mvtec_ad/папка с пробелом/кадр 1.png", "validation")]
    for i, (_, rel, _) in enumerate(rows):
        _png(tmp_path / rel, 100 + i)
    images = pd.DataFrame([dict(image_id=i, relative_path=r, split=s) for i, r, s in rows])
    return tmp_path, images


@pytest.fixture
def pairs():
    def row(pid, q, r, split, gt):
        return dict(pair_id=pid, query_id=q, reference_id=r, relation_id="P2_NO_CHANGE_IN_REGION", y_match=1,
                    split=split, source_group_id="g", transform_id="blur", severity=0.5, seed=None,
                    gt_H_path=gt, gt_mask_path=gt, input_mask_path=None)
    return pd.DataFrame([row("p1", "kadid10k:000000", "mvtec_ad:000000", "validation", "gt/a.txt"),
                         row("p0", "mvtec_ad:000000", "kadid10k:000000", "validation", "gt/b.txt"),
                         row("p2", "kadid10k:000001", "kadid10k:000001", "test", "gt/c.txt")])


def test_inputs_have_no_forbidden_columns(pairs):           # INV-11
    guard = SplitGuard("pilot", fixture=False)
    inputs = load_pair_inputs(pairs, BASE + ["transform_id", "severity"], "validation", guard)
    assert [p.pair_id for p in inputs] == ["p0", "p1"]   # deterministic order
    for p in inputs:
        assert isinstance(p, PairInput) and set(p.extras) == {"transform_id", "severity"}
        assert not {"y_match", "gt_H_path", "gt_mask_path"} & set(p.extras) and not hasattr(p, "y_match")
    for bad in (BASE + ["y_match"], BASE + ["gt_mask_path"], BASE[:3], BASE + ["split"]):
        with pytest.raises(ConfigError):
            load_pair_inputs(pairs, bad, "validation", guard)


def _fake_predictor(inputs, store):
    return [(p.pair_id, float(store.load(p.query_id, "gray").mean() - store.load(p.reference_id, "gray").mean()))
            for p in inputs]


def test_predictions_unchanged_without_gt(pairs, store_data):   # T-DAT-05
    root, images = store_data
    guard = SplitGuard("pilot", fixture=False)
    store = ImageStore(root, images, guard)
    full = _fake_predictor(load_pair_inputs(pairs, BASE, "validation", guard), store)
    stripped = pairs.drop(columns=["gt_H_path", "gt_mask_path"])
    assert _fake_predictor(load_pair_inputs(stripped, BASE, "validation", guard), store) == full
    ev = load_pair_evaluation(pairs, "validation")
    assert {"y_match", "gt_H_path", "gt_mask_path"} <= set(ev.columns) and set(ev.split) == {"validation"}


def test_pilot_cannot_load_test(store_data, pairs):
    root, images = store_data
    guard = SplitGuard("pilot", fixture=False)
    with pytest.raises(TestAccessError):
        ImageStore(root, images, guard).load("kadid10k:000001", "gray")
    with pytest.raises(TestAccessError):
        load_pair_inputs(pairs, BASE, "test", guard)


def test_split_guard_table():   # §7.4
    pilot, smoke = SplitGuard("pilot", fixture=False), SplitGuard("smoke", fixture=True)
    for split in ("train", "dev", "validation"):
        pilot.check(split)
    for split in ("test", "synthetic_test", "real_test"):
        with pytest.raises(TestAccessError):
            pilot.check(split)
        smoke.check(split)
    conf = SplitGuard("confirmatory", fixture=False)
    with pytest.raises(TestAccessError):
        conf.check("test")
    conf.allow_test()
    conf.check("real_test")
    with pytest.raises(ProtocolViolationError):
        pilot.allow_test()
    with pytest.raises(ConfigError):
        SplitGuard("exploratory", fixture=False)


def test_color_required(store_data):
    root, images = store_data
    store = ImageStore(root, images, SplitGuard("smoke", fixture=True))
    with pytest.raises(TypeError):
        store.load("kadid10k:000000")
    with pytest.raises(ConfigError):
        store.load("kadid10k:000000", "hsv")
    bgr, rgb, gray = (store.load("kadid10k:000000", c) for c in ("bgr", "rgb", "gray"))
    assert bgr.shape == (4, 6, 3) and gray.shape == (4, 6) and (rgb == bgr[..., ::-1]).all()


def test_imagestore_loads_path_with_spaces_and_unicode(store_data):   # Review Focus 4
    root, images = store_data
    img = ImageStore(root, images, SplitGuard("pilot", fixture=False)).load("mvtec_ad:000000", "bgr")
    assert img.shape == (4, 6, 3) and int(img[0, 0, 1]) == 102


def test_imagestore_errors_name_only_image_id(store_data):
    root, images = store_data
    store = ImageStore(root, images, SplitGuard("pilot", fixture=False))
    with pytest.raises(DataValidationError, match="unknown"):
        store.load("kadid10k:999999", "gray")
    (root / "kadid10k/images/I01.png").write_bytes(b"broken")
    with pytest.raises(DataValidationError) as e:
        store.load("kadid10k:000000", "gray")
    assert "kadid10k:000000" in str(e.value) and "I01" not in str(e.value)


def test_imagestore_never_exposes_paths(store_data):
    root, images = store_data
    store = ImageStore(root, images, SplitGuard("pilot", fixture=False))
    assert not hasattr(store, "path_of")
    assert not [a for a in dir(store) if not a.startswith("_") and a != "load"]
