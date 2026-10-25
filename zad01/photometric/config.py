"""Загрузка и проверка YAML-конфигурации запуска."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from photometric.methods.registry import DEFAULT_METHOD_IDS, resolve

# Обязательный фотометрический поднабор KADID-10k (ФОТО-01).
REQUIRED_KADID_TYPES = ["07", "08", "10", "11", "12", "13", "14", "16", "17", "18", "25"]

DEFAULTS: dict = {
    "run_name": "p1",
    "output_dir": "outputs/p1",
    "color_profile": "sRGB",
    "kadid": {
        "root": None,
        "images_subdir": "images",
        "dmos_csv": "dmos.csv",
        "types": list(REQUIRED_KADID_TYPES),
        "levels": [1, 2, 3, 4, 5],
        "originals": None,  # None — все 81; список ID (например ["I01", ...]) — урезанный дымовой прогон
        "validate_full": True,
    },
    "split": {"seed": 20261023, "fractions": {"validation": 0.5, "test": 0.5}},
    "pairs": {"seed": 20261024, "negatives_per_query": 3},
    "thresholds": {"alpha": 0.05, "tune_experiment": "P1-E1", "tune_split": "validation"},
    "methods": list(DEFAULT_METHOD_IDS),
    "experiments": {
        "P1-E1": {"enabled": True},
        "P1-E2": {
            "enabled": True,
            "brightness": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "contrast": [0.6, 0.8, 1.0, 1.2, 1.4],
            "gamma": [0.6, 0.8, 1.0, 1.2, 1.6],
            "noise_sigma": [0.0, 0.01, 0.03, 0.05, 0.1],
            "noise_seeds": 5,
            "noise_seed": 20261025,
            "jpeg_quality": [100, 90, 70, 50, 30],
            "jpeg": {"library": "pillow", "subsampling": "4:4:4", "optimize": False},
        },
        "P1-E3": {"enabled": True, "seed": 20261026},
        "P1-E4": {"enabled": True, "n_images": 8, "warmup": 2, "repeats": 10},
    },
    "cache": {"memory_mb": 1536, "persist_pixel_queries": False},
    "plots": True,
}

JPEG_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(path.resolve())
    base = path.resolve().parent.parent if path.resolve().parent.name == "configs" else path.resolve().parent
    cfg["_base_dir"] = str(base)
    validate(cfg)
    return cfg


def resolve_path(cfg: dict, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(cfg.get("_base_dir", ".")) / p


def validate(cfg: dict) -> None:
    k = cfg["kadid"]
    if not k.get("root"):
        raise ValueError("kadid.root must point to the KADID-10k directory")
    types = [f"{int(t):02d}" for t in k["types"]]
    k["types"] = types
    missing = [t for t in REQUIRED_KADID_TYPES if t not in types]
    if missing:
        raise ValueError(f"kadid.types must include the mandatory photometric subset; missing {missing}")
    if sorted(k["levels"]) != [1, 2, 3, 4, 5]:
        raise ValueError("kadid.levels must be [1..5]: hard levels may not be dropped (ПРИЁМ-Ф4)")
    fr = cfg["split"]["fractions"]
    if set(fr) != {"validation", "test"} or abs(sum(fr.values()) - 1.0) > 1e-9 or min(fr.values()) <= 0:
        raise ValueError("split.fractions must be {validation: p, test: 1-p} with p in (0,1)")
    if cfg["pairs"]["negatives_per_query"] != 3:
        raise ValueError("ФОТО-02 fixes 3 negative pairs per query")
    a = cfg["thresholds"]["alpha"]
    if not 0 < a < 1:
        raise ValueError("thresholds.alpha must be in (0,1)")
    for m in cfg["methods"]:
        resolve(m)
    e2 = cfg["experiments"]["P1-E2"]
    if e2["jpeg"]["library"] != "pillow":
        raise ValueError("only the pillow JPEG backend is implemented")
    if e2["jpeg"]["subsampling"] not in JPEG_SUBSAMPLING:
        raise ValueError(f"jpeg.subsampling must be one of {list(JPEG_SUBSAMPLING)}")
    if int(e2["noise_seeds"]) < 1:
        raise ValueError("noise_seeds must be >= 1")


def config_digest(cfg: dict) -> str:
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(public, sort_keys=True, default=str).encode()).hexdigest()
