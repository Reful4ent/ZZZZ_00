"""Tiny SOP-like manifest + in-memory image store for metric smoke runs."""
import zlib

import numpy as np
import pandas as pd


class FakeStore:
    def __init__(self, images: pd.DataFrame) -> None:
        self._hw = dict(zip(images["image_id"], zip(images["height"], images["width"])))

    def load(self, image_id, color="rgb"):
        h, w = self._hw[image_id]
        rng = np.random.default_rng(zlib.crc32(image_id.encode()))
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_sop_like(n_classes=24, per_class=(2, 5), size=(48, 96), seed=0, n_val_classes=6):
    """n_classes train classes + n_val_classes validation classes (disjoint class_id)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes + n_val_classes):
        split = "train" if c < n_classes else "validation"
        for _ in range(int(rng.integers(per_class[0], per_class[1] + 1))):
            iid = f"{len(rows):05d}"
            rows.append({
                "image_id": iid, "dataset_id": "sop_fake", "source_group_id": str(c),
                "class_id": str(c), "category_id": "fake", "split": split,
                "relative_path": f"{iid}.jpg", "sha256": "", "width": int(rng.integers(*size) + 1),
                "height": int(rng.integers(*size) + 1), "channels": 3, "defect_type": None,
            })
    df = pd.DataFrame(rows)
    return df, FakeStore(df)


def smoke_params(lam=0.0, queues=True, epochs=2, steps_per_epoch=3):
    """Spec §11 template with the smoke overrides."""
    return {
        "model": {"backbone": "resnet50", "weights": None, "weights_sha256": None,
                  "embedding_dim": 128, "bn_mode": "frozen_stats_trainable_affine"},
        "input": {"resize_short": 64, "crop": 56, "train_augment": True, "flip_p": 0.5,
                  "normalize_l2": True},
        "batch": {"classes": 16, "per_class": 2},
        "optimizer": {"name": "adam", "lr": 1.0e-4, "beta_1": 0.9, "beta_2": 0.999,
                      "epsilon": 1.0e-7},
        "train": {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "precision": "fp32"},
        "proxy": {"kappa": 16, "init": "normal_then_normalize", "eps_p": 1.0e-12},
        "embed": {"eps_z": 1.0e-12},
        "calibration": {"lambda": lam, "queues": queues, "K": 32, "n_warm": 4},
        "queue": {"check_every": 1000, "sum_tol": 1.0e-6},
        "loss": {"margin": None},
        "data": {"class_subset": None, "per_class_limit": None, "cohort_min_images": None},
        "checkpoint": {"every_steps": None},
        "search": {"block_q": 1024, "block_g": 8192, "k": [1, 5, 10], "d2_tol": 1.0e-6},
        "pairs": {"max_pos_per_class": 20, "max_attempts": 100},
        "threshold": {"criterion_id": "max_recall_at_fpr_le_0.05", "fpr_max": 0.05},
        "degradation": "none",
    }
