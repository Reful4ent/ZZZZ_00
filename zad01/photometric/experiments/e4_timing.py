"""P1-E4: вычислительное время (представление, сравнение, полное)."""

from __future__ import annotations

import pandas as pd

from photometric.timing import benchmark

EXPERIMENT = "P1-E4"


def run(manifest: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    e4 = cfg["experiments"][EXPERIMENT]
    m = manifest[manifest["experiment"] == "P1-E1"]
    if m.empty:
        m = manifest
    # Детерминированная выборка: положительные пары, равномерно по каноническому порядку.
    pos = m[m["label"] == 1].sort_values("pair_id")
    n = int(e4["n_images"])
    sample = pos.iloc[:: max(1, len(pos) // n)].head(n)
    return benchmark(cfg["methods"], sample, int(e4["warmup"]), int(e4["repeats"]), cfg["color_profile"])
