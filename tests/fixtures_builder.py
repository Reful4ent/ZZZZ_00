"""Synthetic datasets in the providers' public layouts, built deterministically from PCG64(7).

Each make_* writes under `root` and returns the counts used as `expected` in fixture configs.
Every image has distinct random content, so no accidental cross-split duplicates arise.
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from expcore.hashing import sha256_file

FIXTURE_SEED = 7
SIZE = 16


_STREAMS = {"kadid10k": 0, "hpatches": 1, "mvtec_ad": 2, "sop": 3}


def _rng(dataset_id: str) -> np.random.Generator:
    """Independent stream per dataset, so images never coincide across datasets."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(FIXTURE_SEED, spawn_key=(_STREAMS[dataset_id],))))


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)   # cv2.imwrite cannot write non-ASCII paths on Windows
    if not ok:
        raise OSError(f"cv2.imencode failed for {path}")
    buf.tofile(path)


def _noise(rng: np.random.Generator, channels: int = 3) -> np.ndarray:
    shape = (SIZE, SIZE, channels) if channels > 1 else (SIZE, SIZE)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def sha_tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def make_kadid(root: Path, n_originals: int = 6, n_types: int = 2, n_levels: int = 2) -> dict[str, int]:
    """images/I01.png and derivatives images/I01_<type>_<level>.png."""
    rng, images = _rng("kadid10k"), Path(root) / "images"
    for o in range(1, n_originals + 1):
        _write(images / f"I{o:02d}.png", _noise(rng))
        for t in range(1, n_types + 1):
            for lv in range(1, n_levels + 1):
                _write(images / f"I{o:02d}_{t:02d}_{lv:02d}.png", _noise(rng))
    return {"originals": n_originals, "derivatives": n_originals * n_types * n_levels}


def make_hpatches(root: Path, n_i: int = 5, n_v: int = 5, complete: bool = True) -> dict[str, int]:
    """<seq>/1.ppm…6.ppm and H_1_2…H_1_6; complete=False drops 6.ppm and H_1_6 from the first v_ sequence."""
    rng, root = _rng("hpatches"), Path(root)
    names = [f"i_seq{k:02d}" for k in range(n_i)] + [f"v_seq{k:02d}" for k in range(n_v)]
    incomplete = None if complete else f"v_seq{0:02d}"
    for name in names:
        last = 5 if name == incomplete else 6
        for k in range(1, last + 1):
            _write(root / name / f"{k}.ppm", _noise(rng))
        for k in range(2, last + 1):
            h = np.eye(3) + rng.normal(0, 1e-3, (3, 3))
            (root / name / f"H_1_{k}").write_text(
                "\n".join(" ".join(f"{v:.10e}" for v in row) for row in h) + "\n", encoding="utf-8")
    return {"sequences": len(names)}


def make_mvtec(root: Path, categories: Sequence[str] = ("carpet", "leather", "tile", "wood"), n_train: int = 8,
               n_test_good: int = 2, n_test_defect: int = 2) -> dict[str, int]:
    """<cat>/train/good, <cat>/test/good, <cat>/test/crack, <cat>/ground_truth/crack/<stem>_mask.png."""
    rng, root = _rng("mvtec_ad"), Path(root)
    for cat in categories:
        for i in range(n_train):
            _write(root / cat / "train" / "good" / f"{i:03d}.png", _noise(rng))
        for i in range(n_test_good):
            _write(root / cat / "test" / "good" / f"{i:03d}.png", _noise(rng))
        for i in range(n_test_defect):
            _write(root / cat / "test" / "crack" / f"{i:03d}.png", _noise(rng))
            mask = np.zeros((SIZE, SIZE), np.uint8)
            mask[4:8, 4:8] = 255
            _write(root / cat / "ground_truth" / "crack" / f"{i:03d}_mask.png", mask)
    n = len(categories)
    return {"train_good": n * n_train, "test_good": n * n_test_good, "test_defect": n * n_test_defect}


def make_sop(root: Path, n_train_classes: int = 10, n_test_classes: int = 4, per_class: int = 3) -> dict[str, int]:
    """Ebay_train.txt / Ebay_test.txt (image_id class_id super_class_id path) and <super>_final/*.JPG."""
    rng, root = _rng("sop"), Path(root)
    supers = ("bicycle", "cabinet")
    image_id = 0
    for split, classes in (("train", range(1, n_train_classes + 1)),
                           ("test", range(n_train_classes + 1, n_train_classes + n_test_classes + 1))):
        lines = ["image_id class_id super_class_id path"]
        for class_id in classes:
            super_id = class_id % len(supers) + 1
            for k in range(per_class):
                image_id += 1
                rel = f"{supers[super_id - 1]}_final/{class_id}_{k}.JPG"
                _write(root / rel, _noise(rng))
                lines.append(f"{image_id} {class_id} {super_id} {rel}")
        (root / f"Ebay_{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"products": n_train_classes + n_test_classes, "train_images": n_train_classes * per_class,
            "test_images": n_test_classes * per_class}


def make_all(data_root: Path) -> dict[str, dict[str, int]]:
    """All four datasets under data_root/<dataset_id>/ (the layout scanners read)."""
    data_root = Path(data_root)
    return {"kadid10k": make_kadid(data_root / "kadid10k"), "hpatches": make_hpatches(data_root / "hpatches"),
            "mvtec_ad": make_mvtec(data_root / "mvtec_ad"), "sop": make_sop(data_root / "sop")}


if __name__ == "__main__":   # python -m tests.fixtures_builder <data_root>
    import sys
    print(make_all(Path(sys.argv[1])))
