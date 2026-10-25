"""Полный прогон P1: манифест → оценки → пороги → решения → метрики → время → кривые."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from photometric import __version__, plots, synthetic
from photometric.cache import RepresentationCache
from photometric.config import config_digest, resolve_path
from photometric.experiments import e1_kadid, e2_synthetic, e3_permutation, e4_timing
from photometric.kadid import load_kadid
from photometric.pairs import manifest_digest
from photometric.scoring import PairScorer, apply_thresholds, method_ids
from photometric.split import split_originals
from photometric.thresholds import select_threshold


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _finite_json(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else None)
    if isinstance(obj, dict):
        return {k: _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_json(v) for v in obj]
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_finite_json(obj), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def build_manifest(cfg: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    k = cfg["kadid"]
    images = load_kadid(resolve_path(cfg, k["root"]), k["images_subdir"], k["dmos_csv"], k["validate_full"])
    originals = images[images["type_id"] == "00"][["original_id", "path"]].sort_values("original_id").reset_index(drop=True)
    if k["originals"] is not None:
        wanted = [str(o) for o in k["originals"]]
        unknown = set(wanted) - set(originals["original_id"])
        if unknown:
            raise ValueError(f"kadid.originals: unknown ids {sorted(unknown)}")
        originals = originals[originals["original_id"].isin(wanted)].reset_index(drop=True)
    split_of = split_originals(originals["original_id"], cfg["split"]["seed"], cfg["split"]["fractions"])
    pd.DataFrame(sorted(split_of.items()), columns=["original_id", "split"]).to_csv(out / "split.csv", index=False)
    images.to_parquet(out / "images.parquet", index=False)

    ex = cfg["experiments"]
    parts = []
    if ex["P1-E1"]["enabled"]:
        parts.append(e1_kadid.build_manifest(images, originals, split_of, cfg))
        log(f"P1-E1 manifest: {len(parts[-1])} pairs")
    if ex["P1-E2"]["enabled"]:
        parts.append(e2_synthetic.build_manifest(originals, split_of, cfg, out / "synthetic" / "P1-E2"))
        log(f"P1-E2 manifest: {len(parts[-1])} pairs")
    if ex["P1-E3"]["enabled"]:
        parts.append(e3_permutation.build_manifest(originals, split_of, cfg, out / "synthetic" / "P1-E3"))
        log(f"P1-E3 manifest: {len(parts[-1])} pairs")
    manifest = pd.concat(parts, ignore_index=True)
    for c in ("level", "noise_seed"):
        manifest[c] = pd.array(manifest[c], dtype="Int64")
    manifest["param"] = pd.to_numeric(manifest["param"], errors="coerce")
    manifest["synthetic"] = manifest["synthetic"].fillna(False).astype(bool)
    manifest["label"] = manifest["label"].astype(np.int8)
    return manifest, images, split_of


def run(cfg: dict) -> Path:
    t_start = time.time()
    out = resolve_path(cfg, cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    log(f"output: {out}")

    manifest, images, split_of = build_manifest(cfg, out)
    digest = manifest_digest(manifest)
    manifest.to_parquet(out / "manifest.parquet", index=False)
    log(f"manifest: {len(manifest)} pairs, sha256={digest[:16]}")

    cache = RepresentationCache(out / "representation_cache", cfg["cache"]["memory_mb"])
    scorer = PairScorer(cfg["methods"], cache, cfg["color_profile"], cfg["cache"]["persist_pixel_queries"])
    step = max(1, manifest["query_id"].nunique() // 20)

    def progress(i, n):
        if i % step == 0 or i == n:
            log(f"scoring queries {i}/{n}")

    t0 = time.time()
    pred = scorer.score_manifest(manifest, progress)
    score_seconds = time.time() - t0
    _check_completeness(pred, manifest, cfg)

    # Пороги: один на метод, по validation основного опыта.
    th_cfg = cfg["thresholds"]
    tune = pred[(pred["experiment"] == th_cfg["tune_experiment"]) & (pred["split"] == th_cfg["tune_split"])]
    if tune.empty:
        raise ValueError(f"no pairs for threshold tuning ({th_cfg['tune_experiment']}, {th_cfg['tune_split']})")
    thresholds = {}
    for mid in method_ids(cfg["methods"]):
        g = tune[tune["method_id"] == mid]
        r = select_threshold(g["score"].to_numpy(), g["label"].to_numpy(), th_cfg["alpha"])
        r.update(method_id=mid, kind=g["kind"].iloc[0], method_config_hash=g["method_config_hash"].iloc[0],
                 alpha=th_cfg["alpha"], tuned_on={"experiment": th_cfg["tune_experiment"], "split": th_cfg["tune_split"]},
                 tie_break="max Recall_all, then min FPR_all, then larger (stricter) threshold")
        if r["fallback_above_max"]:
            log(f"WARNING {mid}: no admissible threshold with FPR<={th_cfg['alpha']}; Recall = 0 ({r['note']})")
        thresholds[mid] = r
    write_json(out / "thresholds.json", thresholds)

    pred = apply_thresholds(pred, thresholds)
    pred.to_parquet(out / "predictions.parquet", index=False)
    log(f"predictions: {len(pred)} rows")

    tables = {}
    ex = cfg["experiments"]
    if ex["P1-E1"]["enabled"]:
        tables.update(e1_kadid.summarize(pred))
    if ex["P1-E2"]["enabled"]:
        tables.update(e2_synthetic.summarize(pred))
    if ex["P1-E3"]["enabled"]:
        tables.update(e3_permutation.summarize(pred))
        tables["e3_histogram_invariance"] = e3_permutation.histogram_invariance(manifest, cfg["color_profile"])
    timing_meta = None
    if ex["P1-E4"]["enabled"]:
        log("P1-E4 timing benchmark")
        tables["e4_timing"], timing_meta = e4_timing.run(manifest, cfg)
        write_json(out / "metrics" / "e4_timing_meta.json", timing_meta)
    for name, df in tables.items():
        df.to_csv(out / "metrics" / f"{name}.csv", index=False)

    if cfg["plots"]:
        curves_dir = out / "curves"
        for split in ("test", "validation"):
            if "e2_curves" in tables:
                plots.e2_curves(tables["e2_curves"], curves_dir / split, split)
            if "e1_by_type_level" in tables:
                plots.e1_by_level(tables["e1_by_type_level"], curves_dir / split, split)

    meta = {
        "program": "P1 photometric",
        "version": __version__,
        "spec": "ТЗ 01, редакция 1.0 от 23.10.2026",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg["_config_path"],
        "config_sha256": config_digest(cfg),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "manifest_sha256": digest,
        "n_pairs": manifest.groupby("experiment").size().to_dict(),
        "n_queries": manifest.groupby("experiment")["query_id"].nunique().to_dict(),
        "split_sizes": pd.Series(split_of).value_counts().to_dict(),
        "methods": {mid: {"kind": g["kind"].iloc[0], "config_hash": g["method_config_hash"].iloc[0]} for mid, g in pred.groupby("method_id")},
        "libraries": _versions(),
        "libjpeg": synthetic.libjpeg_version(),
        "cache_stats": cache.stats,
        "scoring_seconds": round(score_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "undefined_counts": pred[pred["status"] != "OK"].groupby(["method_id", "status"]).size().reset_index().values.tolist(),
        "fallback_thresholds": [m for m, r in thresholds.items() if r["fallback_above_max"]],
    }
    write_json(out / "run_meta.json", meta)
    log(f"done in {meta['total_seconds']} s")
    return out


def _check_completeness(pred: pd.DataFrame, manifest: pd.DataFrame, cfg) -> None:
    """ПРИЁМ-Ф4: каждая пара оценена каждым методом, с одинаковыми ID и разбиением."""
    mids = set(method_ids(cfg["methods"]))
    per_pair = pred.groupby("pair_id")["method_id"].agg(lambda s: frozenset(s))
    if len(per_pair) != len(manifest) or any(v != mids for v in per_pair):
        raise AssertionError("not every pair was scored by every method")
    if pred.groupby("pair_id")["split"].nunique().max() != 1:
        raise AssertionError("pair has inconsistent split between methods")


def _versions() -> dict:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for p in ("numpy", "scipy", "scikit-image", "PyWavelets", "Pillow", "pandas", "pyarrow", "PyYAML", "matplotlib"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out
