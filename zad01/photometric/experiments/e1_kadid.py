"""P1-E1: официальные выбранные типы KADID, все уровни."""

from __future__ import annotations

import pandas as pd

from photometric import metrics
from photometric.kadid import select_queries
from photometric.pairs import build_pairs

EXPERIMENT = "P1-E1"


def build_manifest(images: pd.DataFrame, originals: pd.DataFrame, split_of, cfg) -> pd.DataFrame:
    k = cfg["kadid"]
    q = select_queries(images, k["types"], k["levels"], list(split_of))
    q = q.rename(columns={"image_id": "query_id", "path": "query_path"})
    q["synthetic"] = False
    return build_pairs(q, originals, split_of, cfg["pairs"]["seed"], EXPERIMENT, cfg["pairs"]["negatives_per_query"])


def summarize(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = pred[pred["experiment"] == EXPERIMENT]
    return {
        "e1_overall": metrics.grouped(p, ["split", "method_id"]),
        "e1_by_type": metrics.grouped(p, ["split", "method_id", "type_id", "type_name"]),
        "e1_by_type_level": metrics.grouped(p, ["split", "method_id", "type_id", "type_name", "level"]),
        "e1_macro_original": metrics.macro_by_original(p, ["split", "method_id"]),
        "e1_macro_original_by_type": metrics.macro_by_original(p, ["split", "method_id", "type_id"]),
    }
