"""Integrity and duplicate checks (spec §4.6, ДАН-02): cross-split overlaps block, within-split ones are listed."""

from pathlib import Path

import pandas as pd

from expcore.errors import LeakageError
from expcore.io import dump_json


def _entries(images: pd.DataFrame, by: list[str], keep) -> list[dict]:
    out = []
    for key, g in images.groupby(by, sort=True):
        if keep(g):
            key = key if isinstance(key, tuple) else (key,)
            out.append({**dict(zip(by, key)), "splits": sorted(set(g["split"])),
                        "image_ids": sorted(g["image_id"]), "groups": sorted(set(g["source_group_id"]))})
    return out


def check_integrity(images: pd.DataFrame) -> dict:
    """Overlaps are looked for inside each dataset (images need image_id, dataset_id, source_group_id, split, sha256)."""
    many_splits = lambda g: g["split"].nunique() > 1  # noqa: E731
    cross_sha = _entries(images, ["dataset_id", "sha256"], many_splits)
    cross_groups = [{k: e[k] for k in ("dataset_id", "source_group_id", "splits", "image_ids")}
                    for e in _entries(images, ["dataset_id", "source_group_id"], many_splits)]
    within = [e for split, g in images.groupby("split", sort=True)
              for e in _entries(g, ["dataset_id", "sha256"], lambda g: len(g) > 1)]
    within = [e for e in within if not any(c["dataset_id"] == e["dataset_id"] and c["sha256"] == e["sha256"]
                                           for c in cross_sha)]
    return {"cross_split_sha256": cross_sha, "cross_split_groups": cross_groups,
            "within_split_duplicates": within, "blocking": bool(cross_sha or cross_groups)}


def write_duplicates_report(report: dict, path: Path) -> None:
    dump_json(report, path)


def assert_not_blocking(report: dict) -> None:
    if report["blocking"]:
        raise LeakageError(f"{len(report['cross_split_sha256'])} cross-split identical image(s) and "
                           f"{len(report['cross_split_groups'])} source group(s) split across parts; "
                           "build a new manifest version with overrides (exclude_image)")
