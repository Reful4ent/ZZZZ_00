"""KADID-10k: 81 оригинал, 10 125 производных = 81 × 25 × 5 (ФОТО-01).

Имена файлов: оригинал Ixx.png, производная Ixx_yy_zz.png (xx — изображение,
yy — тип искажения, zz — уровень 1–5). Названия типов читаются из
fixtures/kadid_types.csv, переписанного с официального описания базы.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

N_ORIGINALS = 81
N_TYPES = 25
N_LEVELS = 5
KADID_URL = "https://database.mmsp-kn.de/kadid-10k-database.html"

_DIST_RE = re.compile(r"^I(\d{2})_(\d{2})_(\d{2})\.png$")
_REF_RE = re.compile(r"^I(\d{2})\.png$")

TYPES_FIXTURE = Path(__file__).with_name("data") / "kadid_types.csv"


def load_type_names() -> dict[str, str]:
    with open(TYPES_FIXTURE, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    names = {f"{int(r['type_id']):02d}": r["name"] for r in rows}
    if len(names) != N_TYPES:
        raise ValueError(f"{TYPES_FIXTURE} must list {N_TYPES} types")
    return names


def load_kadid(root, images_subdir: str = "images", dmos_csv: str = "dmos.csv", validate_full: bool = True) -> pd.DataFrame:
    """Таблица всех изображений базы: image_id, original_id, type_id, level, path, dmos.

    Для оригиналов type_id = "00", level = 0.
    """
    root = Path(root)
    img_dir = root / images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"KADID images directory not found: {img_dir}")
    dmos_path = root / dmos_csv
    if not dmos_path.is_file():
        raise FileNotFoundError(f"KADID dmos file not found: {dmos_path}")

    refs, dists = {}, {}
    for p in img_dir.iterdir():
        if m := _DIST_RE.match(p.name):
            dists[p.stem] = (f"I{m[1]}", m[2], int(m[3]), p)
        elif m := _REF_RE.match(p.name):
            refs[p.stem] = p

    dmos = pd.read_csv(dmos_path)
    need_cols = {"dist_img", "ref_img", "dmos"}
    if not need_cols <= set(dmos.columns):
        raise ValueError(f"{dmos_path}: expected columns {sorted(need_cols)}, got {list(dmos.columns)}")
    dmos["image_id"] = dmos["dist_img"].str.replace(".png", "", regex=False)
    dmos_by_id = dmos.set_index("image_id")

    if validate_full:
        _validate_composition(refs, dists, dmos)

    names = load_type_names()
    rows = []
    for rid, p in sorted(refs.items()):
        rows.append(dict(image_id=rid, original_id=rid, type_id="00", type_name="original", level=0, path=str(p), dmos=float("nan")))
    for did, (oid, t, lvl, p) in sorted(dists.items()):
        if oid not in refs:
            raise ValueError(f"distorted image {did} has no reference {oid}.png")
        ref_in_csv = dmos_by_id["ref_img"].get(did)
        if ref_in_csv is not None and ref_in_csv.replace(".png", "") != oid:
            raise ValueError(f"dmos.csv: {did} references {ref_in_csv}, expected {oid}")
        rows.append(
            dict(
                image_id=did,
                original_id=oid,
                type_id=t,
                type_name=names.get(t, "unknown"),
                level=lvl,
                path=str(p),
                dmos=float(dmos_by_id["dmos"].get(did, float("nan"))),
            )
        )
    return pd.DataFrame(rows)


def _validate_composition(refs: dict, dists: dict, dmos: pd.DataFrame) -> None:
    expected_refs = {f"I{i:02d}" for i in range(1, N_ORIGINALS + 1)}
    expected_dists = {
        f"I{i:02d}_{t:02d}_{l:02d}"
        for i in range(1, N_ORIGINALS + 1)
        for t in range(1, N_TYPES + 1)
        for l in range(1, N_LEVELS + 1)
    }
    errors = []
    if set(refs) != expected_refs:
        errors.append(f"references: missing {sorted(expected_refs - set(refs))[:5]}, extra {sorted(set(refs) - expected_refs)[:5]}")
    if set(dists) != expected_dists:
        errors.append(
            f"distorted: {len(dists)} found, missing {sorted(expected_dists - set(dists))[:5]}, "
            f"extra {sorted(set(dists) - expected_dists)[:5]}"
        )
    if set(dmos["image_id"]) != expected_dists:
        errors.append(f"dmos.csv: {len(dmos)} rows do not match the {len(expected_dists)} distorted images")
    if errors:
        raise ValueError("KADID-10k composition check failed (81 + 10125 + dmos.csv expected):\n  " + "\n  ".join(errors))


def select_queries(images: pd.DataFrame, types, levels, originals=None) -> pd.DataFrame:
    """Запросы P1-E1: производные выбранных типов на всех уровнях."""
    q = images[(images["type_id"].isin(types)) & (images["level"].isin(levels))]
    if originals is not None:
        q = q[q["original_id"].isin(originals)]
    return q.sort_values("image_id").reset_index(drop=True)
