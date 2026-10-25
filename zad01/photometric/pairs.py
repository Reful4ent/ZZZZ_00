"""Формирование пар (ФОТО-02).

Положительная пара: запрос и его исходный оригинал. Три отрицательные пары: другие
оригиналы той же части, выбранные PCG64(pair_seed) без повторения reference_id.
Манифест строится один раз и используется всеми методами.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = [
    "pair_id",
    "experiment",
    "split",
    "query_id",
    "query_path",
    "reference_id",
    "reference_path",
    "original_id",
    "label",
    "type_id",
    "type_name",
    "level",
    "synthetic",
    "factor",
    "param",
    "noise_seed",
    "variant",
]


def _experiment_stream(pair_seed: int, experiment: str) -> np.random.Generator:
    # Отдельный поток PCG64 на каждый опыт: пары одного опыта не зависят от состава других.
    ss = np.random.SeedSequence([int(pair_seed), zlib.crc32(experiment.encode())])
    return np.random.Generator(np.random.PCG64(ss))


def build_pairs(
    queries: pd.DataFrame,
    originals: pd.DataFrame,
    split_of: dict[str, str],
    pair_seed: int,
    experiment: str,
    n_negatives: int = 3,
    negatives: bool = True,
) -> pd.DataFrame:
    """queries: столбцы query_id, query_path, original_id (+ любые описательные столбцы манифеста).

    originals: столбцы original_id, path. Возвращает манифест в каноническом порядке.
    """
    ref_path = dict(zip(originals["original_id"], originals["path"]))
    by_part: dict[str, list[str]] = {}
    for oid in sorted(ref_path):
        by_part.setdefault(split_of[oid], []).append(oid)

    rng = _experiment_stream(pair_seed, experiment)
    rows = []
    q = queries.sort_values("query_id", kind="stable").reset_index(drop=True)
    for rec in q.to_dict("records"):
        oid = rec["original_id"]
        part = split_of[oid]
        base = {c: rec.get(c) for c in MANIFEST_COLUMNS if c in rec}
        base.update(experiment=experiment, split=part, original_id=oid)
        label_pos = int(rec.get("label", 1))
        rows.append({**base, "reference_id": oid, "reference_path": ref_path[oid], "label": label_pos})
        if not negatives:
            continue
        pool = [o for o in by_part[part] if o != oid]
        if len(pool) < n_negatives:
            raise ValueError(f"part {part!r} has {len(pool)} other originals, need {n_negatives}")
        chosen = rng.choice(len(pool), size=n_negatives, replace=False)
        for j in chosen:
            rid = pool[int(j)]
            rows.append({**base, "reference_id": rid, "reference_path": ref_path[rid], "label": 0})

    df = pd.DataFrame(rows)
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["pair_id"] = df["experiment"] + ":" + df["query_id"] + ":" + df["reference_id"]
    if df["pair_id"].duplicated().any():
        raise AssertionError("duplicate pair_id in manifest")
    return df[MANIFEST_COLUMNS]


def manifest_digest(manifest: pd.DataFrame) -> str:
    """SHA-256 канонического CSV манифеста без путей (пути зависят от машины)."""
    cols = [c for c in MANIFEST_COLUMNS if not c.endswith("_path")]
    blob = manifest[cols].to_csv(index=False, lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(blob.encode()).hexdigest()
