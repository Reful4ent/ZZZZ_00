"""Dataset scanners over the providers' public layouts (spec §4.2–4.5) and composition checks.

`scan` reads data_root/<dataset_id>/ and returns one raw row per image; relative_path is
relative to data_root with '/' separators. Unexpected layout → DataValidationError.
"""

import re
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import pandas as pd

from expcore.config import DatasetCfg
from expcore.errors import ConfigError, CountMismatchError, DataValidationError, LeakageError

RAW_COLUMNS = ("natural_key", "dataset_id", "source_group_id", "class_id", "category_id", "relative_path",
               "defect_type", "role", "width", "height", "channels")
ALLOWED_CHANNELS = (1, 3)

_KADID_ORIGINAL = re.compile(r"(I\d+)\.png")
_KADID_DERIVATIVE = re.compile(r"(I\d+)_(\d{2})_(\d{2})\.png")
_HP_IMAGES = tuple(f"{k}.ppm" for k in range(1, 7))
_HP_HOMOGRAPHIES = tuple(f"H_1_{k}" for k in range(2, 7))
_HP_PREFIXES = {"i_": "i", "v_": "v"}
_SOP_HEADER = "image_id class_id super_class_id path"
MVTEC_GOOD = "good"


def to_relative_path(path: Path | str, data_root: Path | str) -> str:
    """Path relative to data_root with '/' separators, whatever separators the input used."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    root = PurePosixPath(str(data_root).replace("\\", "/"))
    return p.relative_to(root).as_posix()


def read_image_meta(path: Path) -> tuple[int, int, int]:
    """(width, height, channels); unreadable file or unexpected channels → DataValidationError naming the file."""
    try:
        data = np.fromfile(path, dtype=np.uint8)   # cv2.imread cannot open non-ASCII paths on Windows
    except OSError as e:
        raise DataValidationError(f"{path}: cannot read image file: {e}") from e
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise DataValidationError(f"{path}: not a readable image")
    channels = 1 if img.ndim == 2 else img.shape[2]
    if channels not in ALLOWED_CHANNELS:
        raise DataValidationError(f"{path}: unexpected number of channels {channels} (allowed {ALLOWED_CHANNELS})")
    return img.shape[1], img.shape[0], channels


def _row(path: Path, data_root: Path, dataset_id: str, natural_key: str, group: str, role: str, *,
         class_id: str | None = None, category_id: str | None = None, defect_type: str | None = None) -> dict:
    width, height, channels = read_image_meta(path)
    return {"natural_key": natural_key, "dataset_id": dataset_id, "source_group_id": group, "class_id": class_id,
            "category_id": category_id, "relative_path": to_relative_path(path, data_root),
            "defect_type": defect_type, "role": role, "width": width, "height": height, "channels": channels}


def _require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise DataValidationError(f"{path}: expected {what} directory is missing")
    return path


def _files(directory: Path) -> list[Path]:
    entries = sorted(directory.iterdir())
    dirs = [p.name for p in entries if p.is_dir()]
    if dirs:
        raise DataValidationError(f"{directory}: unexpected subdirectories {dirs}")
    return entries


def _scan_kadid(cfg: DatasetCfg, data_root: Path) -> list[dict]:
    images = _require_dir(data_root / cfg.dataset_id / "images", "KADID images")
    rows, originals = [], set()
    for p in _files(images):
        if m := _KADID_ORIGINAL.fullmatch(p.name):
            originals.add(m[1])
            rows.append(_row(p, data_root, cfg.dataset_id, m[1], m[1], "original"))
        elif m := _KADID_DERIVATIVE.fullmatch(p.name):
            rows.append(_row(p, data_root, cfg.dataset_id, p.stem, m[1], "derivative", category_id=m[2]))
        else:
            raise DataValidationError(f"{p}: unexpected file in KADID images (expected I<nn>.png or I<nn>_<tt>_<ll>.png)")
    orphans = sorted({r["source_group_id"] for r in rows} - originals)
    if orphans:
        raise DataValidationError(f"{images}: derivatives without an original: {orphans}")
    return rows


def _scan_hpatches(cfg: DatasetCfg, data_root: Path) -> list[dict]:
    root = _require_dir(data_root / cfg.dataset_id, "HPatches")
    rows = []
    for seq in sorted(p for p in root.iterdir() if p.is_dir()):
        category = next((c for prefix, c in _HP_PREFIXES.items() if seq.name.startswith(prefix)), None)
        if category is None:
            raise DataValidationError(f"{seq}: HPatches sequence name must start with i_ or v_")
        for p in _files(seq):
            if p.name in _HP_IMAGES:
                rows.append(_row(p, data_root, cfg.dataset_id, f"{seq.name}/{p.name}", seq.name, "sequence_image",
                                 category_id=category))
            elif p.name not in _HP_HOMOGRAPHIES:
                raise DataValidationError(f"{p}: unexpected file in HPatches sequence")
    return rows


def _scan_mvtec(cfg: DatasetCfg, data_root: Path) -> list[dict]:
    if not cfg.categories:
        raise ConfigError(f"datasets.{cfg.dataset_id}.categories: MVTec categories must be listed")
    root = _require_dir(data_root / cfg.dataset_id, "MVTec AD")
    rows = []
    for cat in cfg.categories:
        cat_dir = _require_dir(root / cat, f"MVTec category {cat!r}")
        train = _require_dir(cat_dir / "train", "MVTec train")
        extra = sorted(p.name for p in train.iterdir() if p.name != MVTEC_GOOD)
        if extra:
            raise DataValidationError(f"{train}: only 'good' is expected, found {extra}")
        for part, subdirs in (("train", [train / MVTEC_GOOD]),
                              ("test", sorted(_require_dir(cat_dir / "test", "MVTec test").iterdir()))):
            for sub in subdirs:
                for p in _files(_require_dir(sub, f"MVTec {part}/{sub.name}")):
                    if p.suffix != ".png":
                        raise DataValidationError(f"{p}: unexpected file in MVTec (expected .png)")
                    if sub.name != MVTEC_GOOD:
                        mask = cat_dir / "ground_truth" / sub.name / f"{p.stem}_mask.png"
                        if not mask.is_file():
                            raise DataValidationError(f"{p}: defect image without mask {mask}")
                    key = f"{cat}/{part}/{sub.name}/{p.stem}"   # USER DECISION 2a: group includes the subdirectory
                    rows.append(_row(p, data_root, cfg.dataset_id, key, key,
                                     "train_good" if part == "train" else "test",
                                     category_id=cat, defect_type=sub.name))
    return rows


def _read_sop_list(path: Path) -> list[tuple[int, int, str]]:
    if not path.is_file():
        raise DataValidationError(f"{path}: SOP list file is missing")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != _SOP_HEADER:
        raise DataValidationError(f"{path}: header must be {_SOP_HEADER!r}")
    out = []
    for n, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4 or not all(f.isdigit() for f in fields[:3]):
            raise DataValidationError(f"{path}:{n}: expected '<image_id> <class_id> <super_class_id> <path>'")
        out.append((int(fields[1]), int(fields[2]), fields[3]))
    return out


def _scan_sop(cfg: DatasetCfg, data_root: Path) -> list[dict]:
    root = _require_dir(data_root / cfg.dataset_id, "SOP")
    lists = {role: _read_sop_list(root / f"Ebay_{role}.txt") for role in ("train", "test")}
    shared = sorted({c for c, _, _ in lists["train"]} & {c for c, _, _ in lists["test"]})
    if shared:
        raise LeakageError(f"SOP: class_id present in both official train and test: {shared[:5]}")
    rows = []
    for role, entries in lists.items():
        for class_id, super_id, rel in entries:
            rows.append(_row(root / rel, data_root, cfg.dataset_id, rel, str(class_id), role,
                             class_id=str(class_id), category_id=str(super_id)))
    return rows


_SCANNERS = {"kadid10k": _scan_kadid, "hpatches": _scan_hpatches, "mvtec_ad": _scan_mvtec, "sop": _scan_sop}


def scan(dataset_cfg: DatasetCfg, data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = _SCANNERS[dataset_cfg.dataset_id](dataset_cfg, data_root)
    raw = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    return raw.astype({"width": "int32", "height": "int32", "channels": "int32"})


def _discover(raw: pd.DataFrame, dataset_id: str) -> dict[str, int]:
    role = raw["role"]
    if dataset_id == "kadid10k":
        return {"originals": int((role == "original").sum()), "derivatives": int((role == "derivative").sum())}
    if dataset_id == "hpatches":
        return {"sequences": int(raw["source_group_id"].nunique())}
    if dataset_id == "mvtec_ad":
        test = role == "test"
        return {"train_good": int((role == "train_good").sum()),
                "test_good": int((test & (raw["defect_type"] == MVTEC_GOOD)).sum()),
                "test_defect": int((test & (raw["defect_type"] != MVTEC_GOOD)).sum())}
    return {"products": int(raw["class_id"].nunique()), "train_images": int((role == "train").sum()),
            "test_images": int((role == "test").sum())}


def check_counts(raw: pd.DataFrame, dataset_cfg: DatasetCfg) -> dict[str, Any]:
    """Compare discovered composition with `expected`; mismatch → CountMismatchError (HPatches: reported only)."""
    dataset_id = dataset_cfg.dataset_id
    discovered = _discover(raw, dataset_id)
    expected = {k: v for k, v in dataset_cfg.expected.items() if not (dataset_id == "hpatches" and k == "subset_name")}
    unknown = sorted(set(expected) - set(discovered))
    if unknown:
        raise ConfigError(f"datasets.{dataset_id}.expected: unknown key(s) {unknown}; known {sorted(discovered)}")
    mismatched = {k: (expected[k], discovered[k]) for k in expected if expected[k] != discovered[k]}
    if dataset_id == "kadid10k" and not mismatched and {"originals", "derivatives"} <= set(expected):
        per = expected["derivatives"] // max(expected["originals"], 1)
        counts = raw[raw["role"] == "derivative"].groupby("source_group_id").size()
        uneven = counts[counts != per]
        if not uneven.empty:
            mismatched["derivatives_per_original"] = (per, uneven.to_dict())
    if mismatched and dataset_id != "hpatches":
        raise CountMismatchError(f"{dataset_id}: expected vs discovered differ: {mismatched}")
    return {"discovered": discovered, "expected": dict(dataset_cfg.expected), "matches": not mismatched}


def hpatches_completeness(raw: pd.DataFrame, dataset_cfg: DatasetCfg, data_root: Path) -> dict[str, Any]:
    """full | subset | unverified (spec §4.3); homographies are checked on disk under data_root."""
    root = Path(data_root) / dataset_cfg.dataset_id
    sequences = sorted(raw["source_group_id"].unique())
    present = raw.groupby("source_group_id")["natural_key"].agg(lambda s: {k.split("/", 1)[1] for k in s})
    incomplete = [seq for seq in sequences
                  if not set(_HP_IMAGES) <= present[seq]
                  or not all((root / seq / h).is_file() for h in _HP_HOMOGRAPHIES)]
    expected_n = dataset_cfg.expected.get("sequences")
    if incomplete or (expected_n is not None and expected_n != len(sequences)):
        status = "subset"
    else:
        status = "unverified" if expected_n is None else "full"
    return {"status": status, "subset_name": dataset_cfg.expected.get("subset_name"),
            "sequences": sequences, "incomplete": incomplete}
