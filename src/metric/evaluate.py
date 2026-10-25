"""P3 evaluation: retrieval, pair scores, validation-only threshold, predictions (NumPy + pandas, no TF)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from expcore.errors import ConfigError, ProtocolViolationError
from expcore.io import dump_json, write_parquet
from expcore.registry import Outcome, register_reason_code
from expcore.thresholds import load_threshold, save_threshold, select_threshold
from metric.embed import EmbeddingSet
from metric.pairs import score_pairs
from metric.search import NO_POSITIVE, blocked_topk, no_positive_mask, retrieval_table

register_reason_code(NO_POSITIVE, Outcome.NOT_APPLICABLE)
CRITERION_ID = "max_recall_at_fpr_le_0.05"
_BASE = dict(score_direction="lower_is_match", timing_mode="amortized_batch")


def criterion_max_recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_max: float = 0.05) -> float:
    """Threshold on the unified score s=-d (predict match iff s >= thr).

    Candidates are the unique scores; keep FPR<=fpr_max, maximise recall, tie -> larger thr (smaller tD).
    If no candidate meets the FPR bound, return nextafter(max, inf), i.e. predict nothing.
    """
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    u = np.unique(s)
    tp = len(pos) - np.searchsorted(pos, u, "left")
    fp = len(neg) - np.searchsorted(neg, u, "left")
    fpr = fp / len(neg) if len(neg) else np.zeros(len(u))
    rec = tp / len(pos) if len(pos) else np.zeros(len(u))
    ok = fpr <= fpr_max
    if not ok.any():
        return float(np.nextafter(u.max(), np.inf))
    return float(u[ok & (rec == rec[ok].max())].max())


def _gallery_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def evaluate_retrieval(gallery: EmbeddingSet, queries: EmbeddingSet, classes: pd.Series, params: dict,
                       ids: dict, *, expected_gallery_ids, td: float | None = None,
                       metric: str = "cosine") -> tuple[pd.DataFrame, dict]:
    """ids: {"out_dir", "run_id", "method_id"}. expected_gallery_ids come from the frozen manifest; gallery_manifest.json is a record.
    metric: the method's search metric ("raw_l2" for proxy_softmax_noL2)."""
    out = Path(ids["out_dir"])
    sha = _gallery_sha256(gallery.image_ids)
    if sha != _gallery_sha256(expected_gallery_ids):
        raise ConfigError("gallery image_ids differ from the frozen manifest")
    out.mkdir(parents=True, exist_ok=True)
    dump_json({"split": gallery.split, "n_images": len(gallery.image_ids), "sha256": sha},
              out / "gallery_manifest.json")
    s = params["search"]
    g_ids, q_ids = np.asarray(gallery.image_ids), np.asarray(queries.image_ids)
    g_cls, q_cls = classes.loc[g_ids].to_numpy(), classes.loc[q_ids].to_numpy()
    res = blocked_topk(queries.embeddings, q_ids, gallery.embeddings, g_ids, s["k"], s["block_q"],
                       s["block_g"], s["d2_tol"], metric)
    df, summ = retrieval_table(res, q_cls, g_ids, g_cls, no_positive_mask(q_ids, q_cls, g_ids, g_cls))
    df["y_hat"] = pd.array([int(d <= td) if td is not None and v and not pd.isna(d) else None
                            for v, d in zip(df["valid"], df["d_top1"])], dtype="Int8")
    for i, (col, v) in enumerate((("run_id", ids["run_id"]), ("method_id", ids["method_id"]),
                                  ("split", queries.split), ("degradation_id", queries.degradation_id))):
        df.insert(i, col, v)   # spec 13.2 column order
    summ = {**summ, "gallery_sha256": sha, "split": gallery.split}
    write_parquet(df, out / "retrieval.parquet", nullable=["d_top1", "rank_first_positive", "y_hat"])
    dump_json(summ, out / "retrieval_summary.json")
    return df, summ


def evaluate_pairs(emb: EmbeddingSet, pairs: pd.DataFrame, split: str, params: dict,
                   threshold_path: Path | None, *, ids: dict) -> pd.DataFrame:
    """ids: {"out_dir", "run_id", "method_id", "manifest_hash"}. Validation selects+saves the threshold
    (threshold_path); test only loads it and requires split_used == validation."""
    if emb.split != split or (pairs["split"] != split).any():
        raise ProtocolViolationError(f"embeddings/pairs split does not match {split!r} (INV-10)")
    if threshold_path is None and split != "validation":
        raise ProtocolViolationError("test evaluation requires a validation threshold_path")
    sc = score_pairs(emb.embeddings, emb.image_ids, pairs, params["search"]["d2_tol"])
    sc.insert(0, "run_id", ids["run_id"])
    sc.insert(1, "method_id", ids["method_id"])
    sc.insert(3, "split", pairs["split"].to_numpy())
    if threshold_path is not None:
        if split == "validation":
            val = sc.assign(y_match=pairs["y_match"].to_numpy(), score_raw=sc["d"],
                            score_direction="lower_is_match", outcome=Outcome.DECIDED.value)
            fpr = params["threshold"]["fpr_max"]
            t = select_threshold(val, lambda s, y: criterion_max_recall_at_fpr(s, y, fpr), run_id=ids["run_id"],
                                 method_id=ids["method_id"], criterion_id=CRITERION_ID,
                                 manifest_hash=ids["manifest_hash"])
            save_threshold(t, Path(threshold_path))
        else:
            load_threshold(Path(threshold_path))  # raises unless present and split_used == validation
    out = Path(ids["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(sc, out / "pair_scores.parquet")
    return sc


def pair_prediction_rows(pair_scores: pd.DataFrame, td: float, *, run_id: str, method_id: str,
                         elapsed_ms: float = 0.0) -> pd.DataFrame:
    """Pair rows: y_pred=1[d<=td]. elapsed_ms is the total, spread evenly (amortized_batch)."""
    d = pair_scores["d"].to_numpy(float)
    return pd.DataFrame({"run_id": run_id, "pair_id": pair_scores["pair_id"].to_numpy(), "query_id": None,
                         "method_id": method_id, "score_raw": d, "y_pred": (d <= td).astype(np.int8),
                         "q_valid": True, "reason_code": None, "elapsed_ms": elapsed_ms / max(len(d), 1),
                         "outcome": Outcome.DECIDED.value, **_BASE})


def query_prediction_rows(retrieval: pd.DataFrame, td: float, *, run_id: str, method_id: str,
                          elapsed_ms: float = 0.0) -> pd.DataFrame:
    """Query rows: score_raw=d_top1; NO_POSITIVE queries are not_applicable (nulls + reason_code).
    Decided rows need y_pred in {0,1} (INV-05), hence td is mandatory."""
    v = retrieval["valid"].to_numpy(bool)
    d = retrieval["d_top1"].to_numpy(float)
    y = pd.Series(np.where(v, d <= td, 0), dtype="Int8").where(v, pd.NA).to_numpy()
    return pd.DataFrame({"run_id": run_id, "pair_id": None, "query_id": retrieval["query_id"].to_numpy(),
                         "method_id": method_id, "score_raw": np.where(v, d, np.nan), "y_pred": y, "q_valid": v,
                         "reason_code": np.where(v, None, NO_POSITIVE),
                         "elapsed_ms": np.where(v, elapsed_ms / max(len(v), 1), np.nan),
                         "outcome": np.where(v, Outcome.DECIDED.value, Outcome.NOT_APPLICABLE.value), **_BASE})
