"""P3 evaluation: retrieval, pair scores, validation-only threshold, predictions (NumPy + pandas, no TF)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from expcore.errors import ConfigError, ProtocolViolationError
from expcore.io import dump_json, load_json, write_parquet
from expcore.registry import Outcome, register_reason_code
from expcore.thresholds import load_threshold, save_threshold, select_threshold
from metric.embed import EmbeddingSet
from metric.pairs import score_pairs
from metric.search import NO_POSITIVE, blocked_topk, no_positive_mask, retrieval_table

register_reason_code(NO_POSITIVE, Outcome.NOT_APPLICABLE)
CRITERION_ID = "max_recall_at_fpr_le_0.05"


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
                       ids: dict, *, td: float | None = None) -> tuple[pd.DataFrame, dict]:
    """ids: {"out_dir"}. gallery_manifest.json lives in out_dir; written once, then checked."""
    out = Path(ids["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sha = _gallery_sha256(gallery.image_ids)
    man = out / "gallery_manifest.json"
    if man.exists():
        if load_json(man).get("sha256") != sha:
            raise ConfigError(f"{man}: gallery image_ids differ from the recorded gallery")
    else:
        dump_json({"split": gallery.split, "n_images": len(gallery.image_ids), "sha256": sha}, man)
    s = params["search"]
    g_ids, q_ids = np.asarray(gallery.image_ids), np.asarray(queries.image_ids)
    g_cls, q_cls = classes.loc[g_ids].to_numpy(), classes.loc[q_ids].to_numpy()
    res = blocked_topk(queries.embeddings, q_ids, gallery.embeddings, g_ids, s["k"], s["block_q"],
                       s["block_g"], s["d2_tol"], "cosine")
    df, summ = retrieval_table(res, q_cls, g_ids, g_cls, no_positive_mask(q_ids, q_cls, g_ids, g_cls))
    if td is not None:
        df["y_hat"] = pd.array([None if pd.isna(d) else int(d <= td) for d in df["d_top1"]], dtype="Int8")
    summ = {**summ, "gallery_sha256": sha, "split": gallery.split}
    write_parquet(df, out / "retrieval.parquet", nullable=["d_top1", "rank_first_positive"])
    dump_json(summ, out / "retrieval_summary.json")
    return df, summ


def evaluate_pairs(emb: EmbeddingSet, pairs: pd.DataFrame, split: str, params: dict,
                   threshold_path: Path | None, *, ids: dict) -> pd.DataFrame:
    """ids: {"out_dir", "run_id", "method_id", "manifest_hash"}. Validation selects+saves the threshold
    (threshold_path); test only loads it and requires split_used == validation."""
    sc = score_pairs(emb.embeddings, emb.image_ids, pairs, params["search"]["d2_tol"])
    sc["y_match"] = pairs["y_match"].to_numpy()
    if threshold_path is not None:
        if split == "validation":
            val = sc.assign(split=split, score_raw=sc["d"], score_direction="lower_is_match",
                            outcome=Outcome.DECIDED.value)
            fpr = params["threshold"]["fpr_max"]
            t = select_threshold(val, lambda s, y: criterion_max_recall_at_fpr(s, y, fpr), run_id=ids["run_id"],
                                 method_id=ids["method_id"], criterion_id=CRITERION_ID,
                                 manifest_hash=ids["manifest_hash"])
            save_threshold(t, Path(threshold_path))
        else:
            if load_threshold(Path(threshold_path)).split_used != "validation":
                raise ProtocolViolationError("test threshold must come from validation")
    elif split != "validation":
        raise ProtocolViolationError("test evaluation requires a validation threshold_path")
    out = Path(ids["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(sc, out / "pair_scores.parquet")
    return sc


def predictions_rows(pair_scores: pd.DataFrame, td: float, *, run_id: str, method_id: str,
                     retrieval: pd.DataFrame | None = None, elapsed_ms: float = 0.0) -> pd.DataFrame:
    """Pair rows (+ query rows if `retrieval` given). Decided rows need y_pred in {0,1} (INV-05), so a
    threshold td is mandatory. elapsed_ms is the total, spread evenly (timing_mode amortized_batch)."""
    n = len(pair_scores) + (0 if retrieval is None else len(retrieval))
    per = elapsed_ms / max(n, 1)
    base = dict(run_id=run_id, method_id=method_id, score_direction="lower_is_match", timing_mode="amortized_batch")
    p = pd.DataFrame({**base, "pair_id": pair_scores["pair_id"].to_numpy(), "query_id": None,
                      "score_raw": pair_scores["d"].to_numpy(float),
                      "y_pred": (pair_scores["d"].to_numpy() <= td).astype(np.int8), "q_valid": True,
                      "reason_code": None, "elapsed_ms": per, "outcome": Outcome.DECIDED.value})
    parts = [p]
    if retrieval is not None:
        v = retrieval["valid"].to_numpy(bool)
        d = retrieval["d_top1"].to_numpy(float)
        parts.append(pd.DataFrame({
            **base, "pair_id": None, "query_id": retrieval["query_id"].to_numpy(),
            "score_raw": np.where(v, d, np.nan),
            "y_pred": pd.Series(np.where(v, d <= td, 0), dtype="Int8").where(v, pd.NA).to_numpy(),
            "q_valid": v, "reason_code": np.where(v, None, NO_POSITIVE), "elapsed_ms": np.where(v, per, np.nan),
            "outcome": np.where(v, Outcome.DECIDED.value, Outcome.NOT_APPLICABLE.value)}))
    return pd.concat(parts, ignore_index=True)
