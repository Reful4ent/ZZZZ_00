"""SOP data helpers: leakage check, proxy index, eligibility, subsets (numpy/pandas only)."""
import math

import pandas as pd

from expcore.errors import LeakageError
from expcore.splits.rng import shuffled


def check_disjoint(images: pd.DataFrame) -> None:
    sets = {s: set(g["class_id"]) for s, g in images.groupby("split")}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sets[a] & sets[b]
            if common:
                raise LeakageError(f"class_id shared by {a} and {b}: {sorted(common)[:5]}")


def build_proxy_index(train_images: pd.DataFrame) -> pd.DataFrame:
    counts = train_images.groupby("class_id").size()
    ids = sorted(counts.index, key=int)
    n = [int(counts[c]) for c in ids]
    return pd.DataFrame({
        "proxy_idx": range(len(ids)), "class_id": ids, "n_images": n,
        "sampler_eligible": [k >= 2 for k in n],
    })


def sampler_exclusions(proxy_index: pd.DataFrame) -> pd.DataFrame:
    out = proxy_index.loc[~proxy_index["sampler_eligible"], ["class_id", "n_images"]].copy()
    out["reason"] = "FEWER_THAN_2_IMAGES"
    return out.reset_index(drop=True)


def apply_class_subset(images: pd.DataFrame, n_classes: int, seed: int) -> pd.DataFrame:
    ids = shuffled(sorted({int(c) for c in images["class_id"]}), seed)[:n_classes]
    keep = {str(c) for c in ids}
    return images[images["class_id"].isin(keep)].reset_index(drop=True)


def apply_per_class_limit(train_images: pd.DataFrame, m: int | None, cohort_min: int,
                          seed: int = 2026) -> pd.DataFrame:
    """Cohort = classes with >= cohort_min images; first m of one shuffled list per class."""
    keep: list[str] = []
    for cid, g in train_images.groupby("class_id"):
        if len(g) < cohort_min:
            continue
        order = shuffled(sorted(g["image_id"]), seed + int(cid))
        keep += order if m is None else order[:m]
    return train_images[train_images["image_id"].isin(set(keep))].reset_index(drop=True)


def steps_per_epoch(n_train_images: int, batch: int = 32) -> int:
    return math.ceil(n_train_images / batch)
