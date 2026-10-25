"""P1-E3: перестановка пикселей как контрпример.

Для каждого оригинала:
  * identity — запрос равен оригиналу (контроль; положительная пара + 3 отрицательные);
  * permuted — пиксели оригинала случайно переставлены (RGB-тройки целиком);
    пара (permuted, оригинал) помечена label = 0: общего пространственного сюжета нет.
Ожидание: совместная гистограмма не меняется (SHI = 1, как у identity), то есть
гистограммный метод не различает изображение и его перестановку; попиксельные меры
и хеши это различие видят.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from photometric import metrics, synthetic
from photometric.io import prepare_image
from photometric.pairs import build_pairs
from photometric.representations import joint_rgb_histogram

EXPERIMENT = "P1-E3"


def _perm_rng(seed: int, original_id: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(original_id.lstrip("I"))])))


def build_manifest(originals: pd.DataFrame, split_of, cfg, out_dir) -> pd.DataFrame:
    seed = cfg["experiments"][EXPERIMENT]["seed"]
    paths = dict(zip(originals["original_id"], originals["path"]))
    items = [dict(original_id=o, query_id=f"P_{o}_perm", factor="permutation", param=float(seed), perm_seed=int(seed)) for o in sorted(paths)]
    q = synthetic.materialize(
        items, paths, out_dir, lambda it, img: synthetic.permute_pixels(img, _perm_rng(it["perm_seed"], it["original_id"])), cfg["color_profile"]
    )
    q["variant"] = "permuted"
    q["label"] = 0
    perm = build_pairs(q, originals, split_of, cfg["pairs"]["seed"], EXPERIMENT, negatives=False)

    ident = pd.DataFrame(
        {"query_id": [f"E3_{o}_identity" for o in sorted(paths)], "query_path": [paths[o] for o in sorted(paths)], "original_id": sorted(paths)}
    )
    ident["variant"] = "identity"
    ident["synthetic"] = False
    ident_pairs = build_pairs(ident, originals, split_of, cfg["pairs"]["seed"], EXPERIMENT, cfg["pairs"]["negatives_per_query"])
    return pd.concat([ident_pairs, perm], ignore_index=True)


def histogram_invariance(manifest: pd.DataFrame, color_profile="sRGB") -> pd.DataFrame:
    """Прямая проверка: гистограмма перестановки побитно совпадает с гистограммой оригинала."""
    rows = []
    m = manifest[(manifest["experiment"] == EXPERIMENT) & (manifest["variant"] == "permuted")]
    for rec in m.itertuples(index=False):
        a = prepare_image(rec.query_path, color_profile)
        b = prepare_image(rec.reference_path, color_profile)
        ha, hb = joint_rgb_histogram(a), joint_rgb_histogram(b)
        rows.append(
            {
                "original_id": rec.original_id,
                "hist_equal": bool(np.array_equal(ha, hb)),
                "hist_max_abs_diff": float(np.abs(ha - hb).max()),
                "shi": float(np.minimum(ha, hb).sum()),
                "pixels_changed_fraction": float(np.any(a != b, axis=2).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = pred[pred["experiment"] == EXPERIMENT]
    rows = []
    for mid, g in p.groupby("method_id", sort=True):
        ident_pos = g[(g["variant"] == "identity") & (g["label"] == 1)]
        ident_neg = g[(g["variant"] == "identity") & (g["label"] == 0)]
        perm = g[g["variant"] == "permuted"]
        rows.append(
            {
                "method_id": mid,
                "raw_identity_mean": ident_pos["raw_value"].mean(),
                "raw_permuted_mean": perm["raw_value"].mean(),
                "raw_permuted_min": perm["raw_value"].min(),
                "raw_permuted_max": perm["raw_value"].max(),
                "raw_other_original_mean": ident_neg["raw_value"].mean(),
                "accept_identity": ident_pos["decision"].mean(),
                "accept_permuted": perm["decision"].mean(),
                "accept_other_original": ident_neg["decision"].mean(),
                "coverage_permuted": (perm["status"] == "OK").mean(),
                # Метод «не различает» перестановку, если её оценка совпадает с identity.
                "permuted_indistinguishable_from_identity": bool(
                    np.allclose(perm["raw_value"].to_numpy(), ident_pos.set_index("original_id").loc[perm["original_id"], "raw_value"].to_numpy(), atol=1e-12)
                ),
            }
        )
    return {"e3_summary": pd.DataFrame(rows), "e3_metrics": metrics.grouped(p, ["method_id", "variant"])}
