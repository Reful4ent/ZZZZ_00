"""Blocked exact top-k search and Recall@k (NumPy + pandas only)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from metric.errors import RunStop

NO_POSITIVE = "NOT_APPLICABLE_NO_POSITIVE_IN_GALLERY"


def check_and_clip_d2(d2: np.ndarray, tol: float) -> np.ndarray:
    lo, hi = float(np.min(d2)), float(np.max(d2))
    if lo < -tol or hi > 4.0 + tol:
        raise RunStop("DISTANCE_RANGE_VIOLATION", {"min": lo, "max": hi, "tol": tol})
    return np.clip(d2, 0.0, 4.0)


@dataclass
class SearchResult:
    query_ids: np.ndarray
    top_idx: np.ndarray  # [Q,k] int64, -1 when absent
    top_d2: np.ndarray  # [Q,k] float64, inf when absent


def blocked_topk(zq, q_ids, zg, g_ids, k: int, block_q: int, block_g: int, tol: float,
                 metric: Literal["cosine", "raw_l2"]) -> SearchResult:
    zq = np.asarray(zq, dtype=np.float64)
    zg = np.asarray(zg, dtype=np.float64)
    q_ids, g_ids = np.asarray(q_ids), np.asarray(g_ids)
    nq, ng = len(zq), len(zg)
    # integer ranks give code-point id order for lexsort tie-breaks
    g_rank = np.unique(g_ids, return_inverse=True)[1].astype(np.int64)
    big = ng + 1
    qn, gn = (zq * zq).sum(1), (zg * zg).sum(1)
    out_idx = np.full((nq, k), -1, dtype=np.int64)
    out_d2 = np.full((nq, k), np.inf)
    for qs in range(0, nq, block_q):
        qe = min(qs + block_q, nq)
        cur_d2 = np.full((qe - qs, k), np.inf)
        cur_idx = np.full((qe - qs, k), -1, dtype=np.int64)
        for gs in range(0, ng, block_g):
            ge = min(gs + block_g, ng)
            dot = zq[qs:qe] @ zg[gs:ge].T
            if metric == "cosine":
                d2 = check_and_clip_d2(2.0 - 2.0 * dot, tol)
            else:
                d2 = np.maximum(0.0, qn[qs:qe, None] + gn[None, gs:ge] - 2.0 * dot)
            d2[q_ids[qs:qe, None] == g_ids[None, gs:ge]] = np.inf
            idx = np.broadcast_to(np.arange(gs, ge, dtype=np.int64), d2.shape)
            all_d2 = np.concatenate([cur_d2, d2], axis=1)
            all_idx = np.concatenate([cur_idx, idx], axis=1)
            all_idx = np.where(np.isinf(all_d2), -1, all_idx)
            rank = np.where(all_idx < 0, big, g_rank[np.maximum(all_idx, 0)])
            order = np.lexsort((rank, all_d2), axis=-1)[:, :k]
            cur_d2 = np.take_along_axis(all_d2, order, 1)
            cur_idx = np.take_along_axis(all_idx, order, 1)
        out_d2[qs:qe, : cur_d2.shape[1]] = cur_d2
        out_idx[qs:qe, : cur_idx.shape[1]] = cur_idx
    return SearchResult(q_ids, out_idx, out_d2)


def no_positive_mask(q_ids, q_cls, g_ids, g_cls) -> np.ndarray:
    cnt = Counter(g_cls.tolist())
    pairs = set(zip(g_ids.tolist(), g_cls.tolist()))
    return np.array(
        [cnt[c] - ((i, c) in pairs) == 0 for i, c in zip(q_ids.tolist(), q_cls.tolist())],
        dtype=bool,
    )


def retrieval_table(res: SearchResult, q_cls, g_ids, g_cls, no_pos: np.ndarray,
                    ks=(1, 5, 10)) -> tuple[pd.DataFrame, dict]:
    kmax = max(ks)
    rows = []
    for i, qid in enumerate(res.query_ids):
        idx = res.top_idx[i][res.top_idx[i] >= 0]
        valid = not no_pos[i]
        pos = np.flatnonzero(g_cls[idx[:kmax]] == q_cls[i])
        rank = int(pos[0]) + 1 if len(pos) else None
        row = {
            "query_id": str(qid), "query_class_id": q_cls[i], "valid": valid,
            "reason_code": None if valid else NO_POSITIVE,
            "top_ids": [str(x) for x in g_ids[idx]],
            "top_d": [float(np.sqrt(x)) for x in res.top_d2[i][: len(idx)]],
            "top_class_ids": [str(x) for x in g_cls[idx]],
            "rank_first_positive": rank,
        }
        for k in ks:
            row[f"hit_at_{k}"] = (rank is not None and rank <= k) if valid else None
        row["d_top1"] = row["top_d"][0] if row["top_d"] else None
        rows.append(row)
    cols = ["query_id", "query_class_id", "valid", "reason_code", "top_ids", "top_d",
            "top_class_ids", "rank_first_positive", *[f"hit_at_{k}" for k in ks], "d_top1"]
    df = pd.DataFrame(rows, columns=cols)
    df["rank_first_positive"] = df["rank_first_positive"].astype(object)
    df.loc[df["rank_first_positive"].isna(), "rank_first_positive"] = None
    n_valid = int(df["valid"].sum())
    summ = {f"recall_at_{k}": (int(df.loc[df["valid"], f"hit_at_{k}"].sum()) / n_valid if n_valid else None)
            for k in ks}
    summ.update(
        correct_at_1=int(df.loc[df["valid"], "hit_at_1"].sum()) if 1 in ks else None,
        n_queries=len(df), n_valid=n_valid, n_no_positive=len(df) - n_valid,
    )
    return df, summ
