"""P3 (SOP same-product) pair manifest builder and pair scoring. No TensorFlow."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from expcore.schemas import PAIRS_COLUMNS
from metric.search import check_and_clip_d2

RELATION = "P3_SAME_PRODUCT"


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in PAIRS_COLUMNS})


def _frame(rows: list[tuple], split: str, seed: int) -> pd.DataFrame:
    if not rows:
        return _empty_pairs()
    q, r, y, g = zip(*rows)
    n = len(rows)
    return pd.DataFrame({
        "pair_id": [f"p3:{split}:{i:07d}" for i in range(n)], "query_id": q, "reference_id": r,
        "relation_id": RELATION, "y_match": np.array(y, dtype=np.int8), "split": split,
        "source_group_id": g, "transform_id": None, "severity": None,
        "seed": np.int64(seed), "gt_H_path": None, "gt_mask_path": None, "input_mask_path": None,
    })[list(PAIRS_COLUMNS)]


def _summary(split, n_classes, y, same, shortage=None, skipped=None) -> dict:
    n_neg = int((y == 0).sum())
    n_same = int(same[y == 0].sum())
    return {"split": split, "n_classes": n_classes, "n_pos": int((y == 1).sum()), "n_neg": n_neg,
            "n_neg_same_superclass": n_same, "n_neg_other_superclass": n_neg - n_same,
            "frac_same_superclass": n_same / n_neg if n_neg else None,
            "classes_with_shortage": shortage, "skipped_pairs": skipped}


def build_p3_pairs(images: pd.DataFrame, split: str, seed: int = 2026, max_pos: int = 20,
                   max_attempts: int = 100) -> tuple[pd.DataFrame, dict]:
    rng = np.random.Generator(np.random.PCG64(seed))
    df = images[images["split"] == split]
    by_class = {int(c): sorted(g["image_id"]) for c, g in df.groupby("class_id")}
    cat_of = df.drop_duplicates("class_id").set_index("class_id")["category_id"]
    cat_of = {int(c): v for c, v in cat_of.items()}
    classes = sorted(by_class)
    seen: set[tuple[str, str]] = set()
    rows: list[tuple] = []
    shortage = skipped = 0

    def take(a: str, b: str) -> bool:
        key = tuple(sorted((a, b)))
        if key in seen:
            return False
        seen.add(key)
        return True

    for c in classes:
        ids = by_class[c]
        allp = list(combinations(ids, 2))
        if len(allp) > max_pos:
            allp = [allp[i] for i in sorted(rng.choice(len(allp), max_pos, replace=False))]
        for a, b in allp:
            take(a, b)
            rows.append((a, b, 1, str(c)))
        n_pos = len(allp)
        same_pool = [k for k in classes if k != c and cat_of[k] == cat_of[c]]
        other_pool = [k for k in classes if cat_of[k] != cat_of[c]]
        n_same = n_pos // 2
        class_short = False
        for j in range(n_pos):
            pools = [same_pool, other_pool] if j < n_same else [other_pool, same_pool]
            for pi, pool in enumerate(pools):
                if not pool:
                    continue
                for _ in range(max_attempts):
                    q = ids[rng.integers(len(ids))]
                    cand = by_class[pool[rng.integers(len(pool))]]
                    r = cand[rng.integers(len(cand))]
                    if take(q, r):
                        rows.append((q, r, 0, str(c)))
                        break
                else:
                    continue
                class_short |= pi == 1
                break
            else:
                skipped += 1
                class_short = True
        shortage += class_short

    out = _frame(rows, split, seed)
    y = out["y_match"].to_numpy()
    cat_by_img = df.set_index("image_id")["category_id"]
    same = (out["query_id"].map(cat_by_img) == out["reference_id"].map(cat_by_img)).to_numpy()
    return out, _summary(split, len(classes), y, same, shortage, skipped)


def p3_pair_builder(images: pd.DataFrame, cfg) -> pd.DataFrame:
    sop = images[images["dataset_id"] == "sop"]
    parts = [build_p3_pairs(sop, s)[0] for s in ("validation", "test")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else _empty_pairs()


def pairs_summary(pairs: pd.DataFrame, images: pd.DataFrame) -> dict:
    cat = images.set_index("image_id")["category_id"]
    out = {}
    for split, g in pairs.groupby("split"):
        n_classes = images.loc[images["split"] == split, "class_id"].nunique()
        same = (g["query_id"].map(cat) == g["reference_id"].map(cat)).to_numpy()
        out[split] = _summary(split, n_classes, g["y_match"].to_numpy(), same)
    return out


def score_pairs(emb: np.ndarray, image_ids: np.ndarray, pairs: pd.DataFrame, tol: float) -> pd.DataFrame:
    row = {k: i for i, k in enumerate(image_ids)}
    emb = np.asarray(emb, dtype=np.float64)
    qi = pairs["query_id"].map(row).to_numpy()
    ri = pairs["reference_id"].map(row).to_numpy()
    s = (emb[qi] * emb[ri]).sum(1)
    d2 = check_and_clip_d2(2.0 - 2.0 * s, tol)
    return pd.DataFrame({"pair_id": pairs["pair_id"].to_numpy(), "query_id": pairs["query_id"].to_numpy(),
                         "reference_id": pairs["reference_id"].to_numpy(), "s_cos": s, "d2": d2,
                         "d": np.sqrt(d2)})
