"""Метрики решений: Recall_all, FPR_all, Coverage.

Recall_all = TP / все положительные пары, FPR_all = FP / все отрицательные пары
(undefined-оценки — отказы с решением 0, но учитываются в знаменателях),
Coverage = доля пар с определённой оценкой.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def basic(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy().astype(bool)
    d = df["decision"].to_numpy().astype(bool)
    defined = df["status"].to_numpy() == "OK"
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    s = df["score"].to_numpy(dtype=np.float64)
    return {
        "n_pairs": int(len(df)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "recall_all": (d & y).sum() / n_pos if n_pos else np.nan,
        "fpr_all": (d & ~y).sum() / n_neg if n_neg else np.nan,
        "coverage": defined.mean() if len(df) else np.nan,
        "coverage_pos": defined[y].mean() if n_pos else np.nan,
        "coverage_neg": defined[~y].mean() if n_neg else np.nan,
        "score_pos_mean": np.nanmean(s[y]) if n_pos and np.isfinite(s[y]).any() else np.nan,
        "score_neg_mean": np.nanmean(s[~y]) if n_neg and np.isfinite(s[~y]).any() else np.nan,
    }


def grouped(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **basic(g)})
    return pd.DataFrame(rows)


def macro_by_original(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Макроусреднение: метрики по каждому original_id, затем среднее и std по оригиналам."""
    per = grouped(df, by + ["original_id"])
    metrics = ["recall_all", "fpr_all", "coverage"]
    agg = per.groupby(by, sort=True)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_macro_{s}" for m, s in agg.columns]
    agg["n_originals"] = per.groupby(by, sort=True)["original_id"].nunique()
    return agg.reset_index()
