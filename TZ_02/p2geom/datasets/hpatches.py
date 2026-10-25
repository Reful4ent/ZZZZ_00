"""HPatches (hpatches-sequences-release): последовательности i_* и v_*,
изображения 1…6 (.ppm), матрицы H_1_j.

Для пары E = image1, I = imagej файл H_1_j действует E -> I, рабочее
преобразование I -> E — обратное (Г-1). Направление проверяется
проекцией точек сетки E с локальными патчами (NCC), результат — в manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..features import to_luma_u8
from ..homography import project
from ..io import read_rgb


@dataclass
class HPair:
    pair_id: str
    part: str                    # i | v
    seq: str                     # последовательность запроса
    ref_seq: str                 # последовательность эталона
    j: int
    ref_path: Path               # E
    query_path: Path             # I
    H_gt: np.ndarray | None      # I -> E; None у отрицательной пары
    negative: bool


def list_sequences(root: str | Path) -> list[Path]:
    root = Path(root)
    seqs = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] in ("i_", "v_"))
    if not seqs:
        raise FileNotFoundError(f"no i_*/v_* sequences under {root}")
    return seqs


def _image_path(seq: Path, k: int) -> Path:
    for ext in (".ppm", ".png", ".jpg"):
        p = seq / f"{k}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{seq}/{k}.*")


def read_h_file(path: Path) -> tuple[np.ndarray, dict]:
    text = path.read_text()
    rows = [r.split() for r in text.strip().splitlines() if r.strip()]
    fmt = {"n_rows": len(rows), "n_cols": sorted({len(r) for r in rows}), "bytes": len(text.encode())}
    H = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError(f"{path}: expected finite 3x3 matrix, got shape {H.shape}")
    fmt["det"] = float(np.linalg.det(H))
    return H, fmt


def _patch_ncc(E_gray: np.ndarray, I_gray: np.ndarray, H_EI: np.ndarray, n: int = 16, r: int = 7) -> float:
    """Средний NCC патчей E и их образов в I при отображении H_EI: E -> I."""
    He, We = E_gray.shape
    Hi, Wi = I_gray.shape
    off = np.stack(np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1)), -1).reshape(-1, 2).astype(float)
    cs = np.stack(np.meshgrid(np.linspace(r, We - 1 - r, n), np.linspace(r, He - 1 - r, n)), -1).reshape(-1, 2)
    scores = []
    for c in cs:
        p = c + off
        q, ok = project(H_EI, p)
        if not ok.all() or q[:, 0].min() < 0 or q[:, 1].min() < 0 or q[:, 0].max() > Wi - 1 or q[:, 1].max() > Hi - 1:
            continue
        a = cv2.remap(E_gray, p[:, 0].astype(np.float32)[None], p[:, 1].astype(np.float32)[None],
                      cv2.INTER_LINEAR).ravel()
        b = cv2.remap(I_gray, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None],
                      cv2.INTER_LINEAR).ravel()
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 1e-9:
            scores.append(float((a * b).sum() / den))
    return float(np.mean(scores)) if scores else float("nan")


def check_direction(E: np.ndarray, I: np.ndarray, H_file: np.ndarray, margin: float = 0.05) -> dict:
    """Сравнение гипотез «файл = E -> I» и «файл = I -> E» по NCC патчей.
    SYMMETRIC — H и H⁻¹ совпадают на сетке E (например, единичная H у i_*):
    направление не влияет на проекции."""
    Eg = to_luma_u8(E).astype(np.float32)
    Ig = to_luma_u8(I).astype(np.float32)
    f = _patch_ncc(Eg, Ig, H_file)
    b = _patch_ncc(Eg, Ig, np.linalg.inv(H_file))
    f_ = -1.0 if np.isnan(f) else f
    b_ = -1.0 if np.isnan(b) else b
    He, We = Eg.shape
    g = np.stack(np.meshgrid(np.linspace(0, We - 1, 10), np.linspace(0, He - 1, 10)), -1).reshape(-1, 2)
    sym = np.abs(project(H_file, g)[0] - project(np.linalg.inv(H_file), g)[0]).max() < 1e-6
    if sym:
        verdict = "SYMMETRIC"
    else:
        verdict = "E_TO_I" if f_ > b_ + margin else ("I_TO_E" if b_ > f_ + margin else "UNDECIDED")
    return {"ncc_file_E_to_I": f, "ncc_file_I_to_E": b, "verdict": verdict}


def build_manifest(root: str | Path, verify: bool = True) -> pd.DataFrame:
    rows = []
    for seq in list_sequences(root):
        E = read_rgb(_image_path(seq, 1)) if verify else None
        for j in range(2, 7):
            hp = seq / f"H_1_{j}"
            H, fmt = read_h_file(hp)
            ip = _image_path(seq, j)
            row = {"seq": seq.name, "part": seq.name[0], "j": j, "h_file": str(hp), "image": str(ip),
                   "h_format": json.dumps(fmt), "H_1_j": json.dumps(H.tolist())}
            if verify:
                I = read_rgb(ip)
                row.update(check_direction(E, I, H))
                row.update(size_E=json.dumps([E.shape[1], E.shape[0]]), size_I=json.dumps([I.shape[1], I.shape[0]]))
            rows.append(row)
    return pd.DataFrame(rows)


def default_split(seqs: list[str], val_fraction: float = 0.3, seed: int = 2026) -> dict[str, list[str]]:
    """Временное разбиение по последовательностям (стратифицировано по i_/v_).
    Нормативное разбиение задаёт ТЗ 00 — передавайте его файлом split JSON."""
    rng = np.random.default_rng(seed)
    out = {"validation": [], "test": []}
    for part in ("i", "v"):
        names = sorted(s for s in seqs if s.startswith(part + "_"))
        perm = [names[k] for k in rng.permutation(len(names))]
        n_val = int(round(val_fraction * len(names)))
        out["validation"] += sorted(perm[:n_val])
        out["test"] += sorted(perm[n_val:])
    return out


def load_split(path: str | Path | None, seqs: list[str]) -> dict[str, list[str]]:
    if path is None:
        return default_split(seqs)
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    unknown = {s for v in split.values() for s in v} - set(seqs)
    if unknown:
        raise ValueError(f"split references unknown sequences: {sorted(unknown)[:5]}")
    return split


def make_pairs(root: str | Path, seq_names: list[str], manifest: pd.DataFrame | None = None,
               negatives: bool = True, seed: int = 2026,
               on_direction_mismatch: str = "error") -> list[HPair]:
    """Положительные пары (E=1, I=j) и по одной отрицательной на каждый запрос:
    эталон другой последовательности той же части (i/v) из того же набора, seed=2026."""
    root = Path(root)
    names = sorted(seq_names)
    verdicts = {}
    if manifest is not None and "verdict" in manifest:
        verdicts = {(r.seq, int(r.j)): r.verdict for r in manifest.itertuples()}
    rng = np.random.default_rng(seed)
    pairs: list[HPair] = []
    for name in names:
        seq = root / name
        for j in range(2, 7):
            H_file, _ = read_h_file(seq / f"H_1_{j}")
            v = verdicts.get((name, j), "E_TO_I")
            if v == "I_TO_E":
                if on_direction_mismatch == "error":
                    raise ValueError(f"{name} H_1_{j}: direction check says I->E")
                H_file = np.linalg.inv(H_file)
            pairs.append(HPair(f"{name}_1_{j}", name[0], name, name, j, _image_path(seq, 1),
                               _image_path(seq, j), np.linalg.inv(H_file), False))
            if negatives:
                others = [s for s in names if s[0] == name[0] and s != name]
                if others:
                    ref = others[int(rng.integers(len(others)))]
                    pairs.append(HPair(f"neg_{name}_{j}_ref_{ref}", name[0], name, ref, j,
                                       _image_path(root / ref, 1), _image_path(seq, j), None, True))
    return pairs
