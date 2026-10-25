"""P1-E4: время построения представления, сравнения и полное время на пару.

Правила ТЗ 04 в распоряжении нет; используется perf_counter_ns с прогревом и
повторами, протокол записывается в результат и может быть заменён.
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np
import pandas as pd

from photometric.io import prepare_image
from photometric.methods import compare, represent, resolve


def _stats(prefix: str, ns: list[int]) -> dict:
    a = np.asarray(ns, dtype=np.float64) / 1e6
    return {
        f"{prefix}_ms_median": float(np.median(a)),
        f"{prefix}_ms_mean": float(a.mean()),
        f"{prefix}_ms_p10": float(np.percentile(a, 10)),
        f"{prefix}_ms_p90": float(np.percentile(a, 90)),
        f"{prefix}_n": int(a.size),
    }


def benchmark(methods, pairs: pd.DataFrame, warmup: int, repeats: int, color_profile="sRGB") -> tuple[pd.DataFrame, dict]:
    """pairs: столбцы query_path, reference_path."""
    decode_ns = []
    imgs = []
    for rec in pairs.itertuples(index=False):
        pair_imgs = []
        for p in (rec.query_path, rec.reference_path):
            t0 = time.perf_counter_ns()
            pair_imgs.append(prepare_image(p, color_profile))
            decode_ns.append(time.perf_counter_ns() - t0)
        imgs.append(pair_imgs)

    rows = []
    for m in methods:
        spec, _, _ = resolve(m)
        rep_ns, cmp_ns, full_ns = [], [], []
        for qi, ri in imgs:
            for _ in range(warmup):
                compare(represent(qi, m), represent(ri, m))
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                rq = represent(qi, m)
                t1 = time.perf_counter_ns()
                rr = represent(ri, m)
                t2 = time.perf_counter_ns()
                compare(rq, rr)
                t3 = time.perf_counter_ns()
                rep_ns += [t1 - t0, t2 - t1]
                cmp_ns.append(t3 - t2)
                full_ns.append(t3 - t0)
        rows.append({"method_id": spec.method_id, **_stats("represent", rep_ns), **_stats("compare", cmp_ns), **_stats("full_pair", full_ns)})
    meta = {
        "protocol": "perf_counter_ns; per pair: warmup x (represent q, represent r, compare), then repeats timed; "
        "full_pair = represent(query) + represent(reference) + compare in one timed block, decoded images in memory",
        "warmup": warmup,
        "repeats": repeats,
        "n_pairs": len(imgs),
        "decode": _stats("decode", decode_ns),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "note": "ТЗ 04 timing rules were not available; replace this protocol when they are fixed",
    }
    return pd.DataFrame(rows), meta
