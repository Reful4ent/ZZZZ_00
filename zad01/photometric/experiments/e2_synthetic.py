"""P1-E2: управляемые одиночные факторы по проектной сетке (ФОТО-03)."""

from __future__ import annotations

import pandas as pd

from photometric import metrics, synthetic
from photometric.pairs import build_pairs

EXPERIMENT = "P1-E2"


def build_manifest(originals: pd.DataFrame, split_of, cfg, out_dir) -> pd.DataFrame:
    e2 = cfg["experiments"][EXPERIMENT]
    items = synthetic.e2_plan(originals["original_id"], e2)
    paths = dict(zip(originals["original_id"], originals["path"]))
    q = synthetic.materialize(items, paths, out_dir, lambda it, img: synthetic.render_e2(it, img, e2), cfg["color_profile"])
    if "noise_seed" not in q.columns:
        q["noise_seed"] = None
    q.to_csv(out_dir / "synthetic_params.csv", index=False)
    return build_pairs(q, originals, split_of, cfg["pairs"]["seed"], EXPERIMENT, cfg["pairs"]["negatives_per_query"])


def summarize(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = pred[pred["experiment"] == EXPERIMENT].copy()
    p["noise_seed"] = p["noise_seed"].fillna(-1).astype(int)
    raw = metrics.grouped(p, ["split", "method_id", "factor", "param", "noise_seed"])
    cols = ["recall_all", "fpr_all", "coverage", "score_pos_mean", "score_neg_mean"]
    agg = raw.groupby(["split", "method_id", "factor", "param"], sort=True)[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = raw.groupby(["split", "method_id", "factor", "param"], sort=True)["noise_seed"].nunique()
    raw["noise_seed"] = raw["noise_seed"].replace(-1, pd.NA)
    return {"e2_curves_per_seed": raw, "e2_curves": agg.reset_index()}
