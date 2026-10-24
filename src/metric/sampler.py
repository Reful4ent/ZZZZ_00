"""Deterministic batch sampler (spec 5.2): batch(t) is a pure function of (seed, t, manifest)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from expcore.seeding import stream
from metric.errors import RunStop


@dataclass
class BatchSpec:
    step: int
    image_ids: list[str]
    class_ids: list[str]
    proxy_idx: np.ndarray
    crop_yx: np.ndarray
    flip: np.ndarray


class BatchSampler:
    def __init__(self, proxy_index: pd.DataFrame, train_images: pd.DataFrame, seed: int,
                 classes: int, per_class: int, resize_short: int, crop: int,
                 train_augment: bool) -> None:
        elig = proxy_index[proxy_index["sampler_eligible"]]
        if len(elig) < classes:
            raise RunStop("SAMPLER_ERROR", {"eligible": len(elig), "classes": classes})
        self.seed, self.classes, self.per_class = seed, classes, per_class
        self.resize_short, self.crop, self.train_augment = resize_short, crop, train_augment
        self._cids = list(elig["class_id"])
        self._pidx = dict(zip(proxy_index["class_id"], proxy_index["proxy_idx"]))
        df = train_images.sort_values("image_id")
        self._imgs = {c: g["image_id"].tolist() for c, g in df.groupby("class_id")}
        self._wh = dict(zip(df["image_id"], zip(df["width"], df["height"])))

    def _resized(self, image_id: str) -> tuple[int, int]:
        """(H', W') after aspect-preserving resize of the short side."""
        w, h = self._wh[image_id]
        s = self.resize_short / min(w, h)
        return (self.resize_short, round(w * s)) if h <= w else (round(h * s), self.resize_short)

    def batch(self, step: int) -> BatchSpec:
        rng = stream(self.seed, "sampler", step)
        drawn = rng.choice(len(self._cids), size=self.classes, replace=False)
        cls, imgs = [], []
        for i in drawn:
            c = self._cids[int(i)]
            pool = self._imgs[c]
            for j in rng.choice(len(pool), size=self.per_class, replace=False):
                cls.append(c)
                imgs.append(pool[int(j)])
        yx = np.zeros((len(imgs), 2), dtype=np.int64)
        flip = np.zeros(len(imgs), dtype=bool)
        aug = stream(self.seed, "augment", step) if self.train_augment else None
        for pos, iid in enumerate(imgs):
            h, w = self._resized(iid)
            if aug is None:
                yx[pos] = ((h - self.crop) // 2, (w - self.crop) // 2)
            else:
                yx[pos] = (aug.integers(0, h - self.crop + 1), aug.integers(0, w - self.crop + 1))
                flip[pos] = aug.random() < 0.5
        return BatchSpec(step, imgs, cls, np.array([self._pidx[c] for c in cls], dtype=np.int32),
                         yx, flip)

    def to_frame(self, spec: BatchSpec) -> pd.DataFrame:
        n = len(spec.image_ids)
        return pd.DataFrame({
            "global_step": spec.step, "pos": range(n), "image_id": spec.image_ids,
            "class_id": spec.class_ids, "proxy_idx": spec.proxy_idx,
            "crop_y": spec.crop_yx[:, 0], "crop_x": spec.crop_yx[:, 1], "flip": spec.flip,
        })
