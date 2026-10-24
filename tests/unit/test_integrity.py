import pandas as pd
import pytest

from expcore.errors import LeakageError
from expcore.integrity import assert_not_blocking, check_integrity, write_duplicates_report
from expcore.io import load_json

pytestmark = pytest.mark.unit   # T-DAT-06


def images(rows):
    return pd.DataFrame([dict(image_id=f"kadid10k:{i:06d}", dataset_id=d, source_group_id=g, split=s, sha256=h)
                         for i, (d, g, s, h) in enumerate(rows)])


CLEAN = [("kadid10k", "I01", "validation", "a" * 64), ("kadid10k", "I02", "test", "b" * 64)]


def test_clean_images():
    rep = check_integrity(images(CLEAN))
    assert rep == {"cross_split_sha256": [], "cross_split_groups": [], "within_split_duplicates": [], "blocking": False}


def test_cross_split_sha_detected():
    rep = check_integrity(images(CLEAN + [("kadid10k", "I03", "test", "a" * 64)]))
    assert rep["blocking"] is True
    (hit,) = rep["cross_split_sha256"]
    assert hit["sha256"] == "a" * 64 and hit["splits"] == ["test", "validation"]
    assert hit["image_ids"] == ["kadid10k:000000", "kadid10k:000002"] and hit["groups"] == ["I01", "I03"]


def test_same_sha_in_different_datasets_is_not_cross_split():
    rep = check_integrity(images(CLEAN + [("sop", "1", "test", "a" * 64)]))
    assert rep["cross_split_sha256"] == [] and rep["blocking"] is False


def test_cross_split_group_detected():
    rep = check_integrity(images(CLEAN + [("kadid10k", "I01", "test", "c" * 64)]))
    assert rep["blocking"] is True
    assert rep["cross_split_groups"] == [{"dataset_id": "kadid10k", "source_group_id": "I01",
                                          "splits": ["test", "validation"],
                                          "image_ids": ["kadid10k:000000", "kadid10k:000002"]}]


def test_within_split_duplicate_not_blocking():
    rep = check_integrity(images(CLEAN + [("kadid10k", "I02", "test", "b" * 64)]))
    assert rep["blocking"] is False and len(rep["within_split_duplicates"]) == 1
    assert rep["within_split_duplicates"][0]["splits"] == ["test"]


def test_assert_not_blocking_raises(tmp_path):
    rep = check_integrity(images(CLEAN + [("kadid10k", "I03", "test", "a" * 64)]))
    write_duplicates_report(rep, tmp_path / "duplicates_report.json")
    assert load_json(tmp_path / "duplicates_report.json") == rep
    with pytest.raises(LeakageError, match="1 cross-split"):
        assert_not_blocking(rep)
    assert_not_blocking(check_integrity(images(CLEAN)))
