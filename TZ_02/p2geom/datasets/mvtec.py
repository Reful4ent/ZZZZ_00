"""MVTec AD: плоские цветные категории, synthetic_test (ОБЛ-01, ОБЛ-02) и
real_test (ОБЛ-03, multi-reference_inspection).

Разбиение train/good по умолчанию (до получения нормативного ТЗ 00):
фиксированное перемешивание seed=2026; первые 5 — эталонная галерея; остаток
делится на три равные части: normal_validation, synthetic_validation
(оригиналы E), synthetic_test (оригиналы E). Каждое E и все производные
остаются в одном split. Разбиение можно передать JSON-файлом.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_rgb, write_png
from ..synth import (SYNTH_AREA_RATIO_RANGE, apply_defect, case_matrix, defect_specs, factor_cases, observe)

MAIN_CATEGORIES = ("carpet", "leather", "tile", "wood")
SPLITS = ("gallery", "normal_validation", "synthetic_validation", "synthetic_test")


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp")) if d.exists() else []


@dataclass
class TestImage:
    path: Path
    defect_type: str            # good | <тип дефекта>
    mask_path: Path | None      # только для оценщика


def scan_category(root: str | Path, category: str) -> tuple[list[Path], list[TestImage]]:
    base = Path(root) / category
    train = _images(base / "train" / "good")
    if not train:
        raise FileNotFoundError(f"{base}/train/good is empty or missing")
    tests = []
    for d in sorted((base / "test").iterdir()):
        for p in _images(d):
            m = base / "ground_truth" / d.name / f"{p.stem}_mask.png"
            tests.append(TestImage(p, d.name, m if m.exists() else None))
    return train, tests


def default_split(train: list[Path], seed: int = 2026, n_gallery: int = 5) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    names = [p.name for p in train]
    perm = [names[k] for k in rng.permutation(len(names))]
    rest = perm[n_gallery:]
    third = len(rest) // 3
    return {"gallery": perm[:n_gallery], "normal_validation": rest[:third],
            "synthetic_validation": rest[third:2 * third], "synthetic_test": rest[2 * third:]}


def load_split(path: str | Path | None, category: str, train: list[Path]) -> dict[str, list[str]]:
    if path is None:
        return default_split(train)
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    return split[category] if category in split else split


def build_synthetic(root: str | Path, out_dir: str | Path, category: str, split_name: str,
                    split: dict[str, list[str]] | None = None, variants_per_original: int = 2,
                    max_originals: int | None = None, seed: int = 2026,
                    min_defect_visible: float | None = 0.5) -> pd.DataFrame:
    """Для каждого оригинала E: variants_per_original пар (I+ = T(E), I− = T(O(E))) с
    одинаковым T (геометрия, фотометрия, окклюдер) и одинаковым PNG-сохранением.
    Спецификации T и O перебираются циклически по сеткам. min_defect_visible=None —
    опыт скрытых дефектов (ограничение видимости снято).
    Возвращает manifest; отклонённые генератором случаи — в generation_report.csv."""
    root, out = Path(root), Path(out_dir) / split_name / category
    out.mkdir(parents=True, exist_ok=True)
    train, _ = scan_category(root, category)
    split = split or default_split(train)
    by_name = {p.name: p for p in train}
    originals = split[split_name][:max_originals] if max_originals else split[split_name]
    donors = [n for n in split[split_name]]
    # Циклический перебор по фиксированным перестановкам: малые наборы покрывают
    # разные случаи, большие — сбалансированы по сеткам T и O.
    perm_rng = np.random.default_rng(seed)
    cases = factor_cases(seed)
    cases = [cases[i] for i in perm_rng.permutation(len(cases))]
    specs = defect_specs()
    specs = [specs[i] for i in perm_rng.permutation(len(specs))]
    rows, report = [], []
    k = 0
    for oi, name in enumerate(originals):
        E = read_rgb(by_name[name])
        size = (E.shape[1], E.shape[0])
        write_png(out / f"{Path(name).stem}_E.png", E)
        for v in range(variants_per_original):
            case, spec = cases[k % len(cases)], specs[k % len(specs)]
            k += 1
            sid = f"{category}_{Path(name).stem}_v{v}"
            rng = np.random.default_rng([seed, oi, v])
            donor = None
            if spec["type"] == "paste":
                cand = [d for d in donors if d != name]
                donor = read_rgb(by_name[cand[int(rng.integers(len(cand)))]])
            O_E, mask, dinfo = apply_defect(E, spec, rng, donor)
            G, attempts = case_matrix(case, size, rng)
            neg = observe(O_E, G, rng, brightness=case["brightness"], occlusion=case["occlusion"],
                          defect_mask_E=mask, min_defect_visible=min_defect_visible)
            if min_defect_visible is not None and (neg.defect_visible_frac or 0.0) < min_defect_visible:
                report.append({"sample_id": sid, "reason": "DEFECT_NOT_VISIBLE",
                               "defect_visible_frac": neg.defect_visible_frac, "case_id": case["case_id"],
                               **{f"defect_{kk}": vv for kk, vv in spec.items()}})
                continue
            pos = observe(E, G, rng, brightness=case["brightness"], occ_I=neg.occ_I, occ_meta=neg.meta)
            write_png(out / f"{sid}_pos.png", pos.I)
            write_png(out / f"{sid}_neg.png", neg.I)
            write_png(out / f"{sid}_visible.png", neg.visible_I)
            write_png(out / f"{sid}_change_mask_E.png", mask)
            common = {
                "sample_id": sid, "category": category, "split": split_name, "original": name,
                "E_path": str(out / f"{Path(name).stem}_E.png"), "visible_path": str(out / f"{sid}_visible.png"),
                "change_mask_path": str(out / f"{sid}_change_mask_E.png"),
                "case_id": case["case_id"], "factor": case["factor"], "factor_value": str(case["value"]),
                "G": json.dumps(G.tolist()), "H_gt": json.dumps(np.linalg.inv(G).tolist()),
                "geom_attempts": attempts, "visibility": neg.visibility,
                "brightness": case["brightness"], "occlusion": case["occlusion"],
                "occ_area_px": neg.meta.get("occ_area_px"), "occ_resamples": neg.meta.get("occ_resamples"),
                "area_ratio_range": json.dumps(SYNTH_AREA_RATIO_RANGE),
            }
            rows.append({**common, "pair_id": sid + "_pos", "I_path": str(out / f"{sid}_pos.png"),
                         "y_match": 1, "defect_type": "none"})
            rows.append({**common, "pair_id": sid + "_neg", "I_path": str(out / f"{sid}_neg.png"),
                         "y_match": 0, "defect_type": spec["type"], "defect_area": spec["area"],
                         "defect_channel": spec.get("channel"), "defect_shift": spec.get("shift"),
                         "defect_area_px": dinfo["actual_area_px"], "defect_area_frac": dinfo["actual_area_frac"],
                         "defect_dE00": dinfo["mean_dE00"], "defect_visible_frac": neg.defect_visible_frac})
    man = pd.DataFrame(rows)
    man.to_parquet(out / "manifest.parquet", index=False)
    pd.DataFrame(report).to_csv(out / "generation_report.csv", index=False)
    return man


def real_test_pairs(root: str | Path, category: str, split: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Все test-запросы × 5 эталонов галереи. Маска дефекта — только для оценщика."""
    train, tests = scan_category(root, category)
    split = split or default_split(train)
    by_name = {p.name: p for p in train}
    rows = []
    for t in tests:
        qid = f"{category}_{t.defect_type}_{t.path.stem}"
        for ref in split["gallery"]:
            rows.append({"pair_id": f"{qid}__ref_{Path(ref).stem}", "query_id": qid, "reference_id": ref,
                         "category": category, "E_path": str(by_name[ref]), "I_path": str(t.path),
                         "defect_type": t.defect_type, "y_match": int(t.defect_type == "good"),
                         "mask_path": str(t.mask_path) if t.mask_path else None})
    return pd.DataFrame(rows)


def normal_validation_pairs(root: str | Path, category: str, split: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """normal validation против dev-галереи (калибровка порогов, y_match=1)."""
    train, _ = scan_category(root, category)
    split = split or default_split(train)
    by_name = {p.name: p for p in train}
    rows = []
    for q in split["normal_validation"]:
        qid = f"{category}_nval_{Path(q).stem}"
        for ref in split["gallery"]:
            rows.append({"pair_id": f"{qid}__ref_{Path(ref).stem}", "query_id": qid, "reference_id": ref,
                         "category": category, "E_path": str(by_name[ref]), "I_path": str(by_name[q]),
                         "defect_type": "good", "y_match": 1, "mask_path": None})
    return pd.DataFrame(rows)


def aggregate_multi_reference(pred: pd.DataFrame) -> pd.DataFrame:
    """Итог по запросу (ОБЛ-03): эталон с минимальным D среди q_cmp=1, при
    равенстве — меньший reference_id; если все пары отвергнуты — reject.
    Выбор эталона не зависит от цветовых порогов, поэтому выбранная строка
    годится и для калибровки. Возвращает по одной строке на query_id."""
    rows = []
    for qid, g in pred.groupby("query_id", sort=True):
        ok = g[g["q_cmp"] == 1].sort_values(["D_glob", "reference_id"], kind="stable")
        row = (ok.iloc[0] if len(ok) else g.sort_values("reference_id", kind="stable").iloc[0]).copy()
        row["n_refs"], row["n_refs_qcmp"] = len(g), len(ok)
        if not len(ok):
            row["y_pred"], row["q_cmp"], row["reference_id"] = np.nan, 0, None
        rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["mode"] = "multi-reference_inspection"
    return out
