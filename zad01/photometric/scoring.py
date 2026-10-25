"""Оценка пар (Ф-5): s = −D для расстояний, s = S для сходств.

Undefined-оценка остаётся отказом (score = NaN, status = код), а не отрицательным score.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from photometric.cache import LazyImage, RepresentationCache
from photometric.methods import DISTANCE, METHODS, compare, resolve


def to_score(kind: str, value: float, status: str) -> float:
    if status != "OK":
        return math.nan
    return -value if kind == DISTANCE else value


def method_ids(methods) -> list[str]:
    return [m if isinstance(m, str) else m["method_id"] for m in methods]


class PairScorer:
    def __init__(self, methods, cache: RepresentationCache, color_profile="sRGB", persist_pixel_queries=False):
        self.methods = list(methods)
        self.cache = cache
        self.color_profile = color_profile
        self.persist_pixel_queries = persist_pixel_queries

    def score_manifest(self, manifest: pd.DataFrame, progress=None) -> pd.DataFrame:
        """Оценивает каждую пару манифеста каждым методом. Одна строка = (pair_id, method_id)."""
        rows = []
        refs: dict[str, LazyImage] = {}
        resolved = [(m, *resolve(m)) for m in self.methods]
        by_query: dict[str, list[dict]] = {}
        cols = ["pair_id", "query_id", "query_path", "reference_id", "reference_path"]
        for rec in manifest[cols].to_dict("records"):
            by_query.setdefault(rec["query_id"], []).append(rec)
        n = len(by_query)
        for gi, qid in enumerate(sorted(by_query)):
            g = by_query[qid]
            qimg = LazyImage(g[0]["query_path"], self.color_profile)
            for m, spec, _, chash in resolved:
                persist_q = self.persist_pixel_queries or not spec.pixel_level
                qrep, q_ns = self.cache.get(m, qid, qimg, persist=persist_q)
                for pair in g:
                    rid = pair["reference_id"]
                    rimg = refs.get(rid) or refs.setdefault(rid, LazyImage(pair["reference_path"], self.color_profile))
                    rrep, r_ns = self.cache.get(m, rid, rimg, persist=True)
                    t0 = time.perf_counter_ns()
                    res = compare(qrep, rrep)
                    cmp_ns = time.perf_counter_ns() - t0
                    rows.append(
                        (pair["pair_id"], spec.method_id, chash, spec.kind, res.value, res.status,
                         to_score(spec.kind, res.value, res.status), q_ns, r_ns, cmp_ns)
                    )
            self.cache.forget(qid)
            qimg.release()
            if progress:
                progress(gi + 1, n)
        out = pd.DataFrame(
            rows,
            columns=["pair_id", "method_id", "method_config_hash", "kind", "raw_value", "status", "score",
                     "repr_query_ns", "repr_reference_ns", "compare_ns"],
        )
        return manifest.merge(out, on="pair_id", how="right", validate="one_to_many")


def score_pair(pair: dict, config, cache: RepresentationCache | None = None) -> list[dict]:
    """Оценивает одну пару всеми методами конфигурации.

    pair: {"query_path", "reference_path"} и необязательные query_id / reference_id / pair_id.
    config: полный конфиг запуска (dict с ключом "methods") или список method_config.
    """
    methods = config["methods"] if isinstance(config, dict) else list(config)
    color_profile = config.get("color_profile", "sRGB") if isinstance(config, dict) else "sRGB"
    qid = pair.get("query_id", str(pair["query_path"]))
    rid = pair.get("reference_id", str(pair["reference_path"]))
    manifest = pd.DataFrame(
        [
            {
                "pair_id": pair.get("pair_id", f"{qid}:{rid}"),
                "query_id": qid,
                "query_path": str(pair["query_path"]),
                "reference_id": rid,
                "reference_path": str(pair["reference_path"]),
            }
        ]
    )
    scorer = PairScorer(methods, cache or RepresentationCache(None), color_profile)
    return scorer.score_manifest(manifest).to_dict("records")


def kinds(methods) -> dict[str, str]:
    return {mid: METHODS[mid].kind for mid in method_ids(methods)}


def apply_thresholds(pred: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """y = 1[s ≥ t]; undefined → решение 0 (отказ)."""
    pred = pred.copy()
    t = pred["method_id"].map({m: v["threshold"] for m, v in thresholds.items()}).astype(np.float64)
    pred["threshold"] = t
    s = pred["score"].to_numpy(dtype=np.float64)
    pred["decision"] = (np.isfinite(s) & (s >= t.to_numpy())).astype(np.int8)
    return pred
