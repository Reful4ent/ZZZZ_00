"""Сквозной smoke-прогон без внешних датасетов.

Строит миниатюрные наборы в формате HPatches и MVTec AD из встроенных
изображений skimage и проходит весь протокол: manifest -> P2-E1 -> калибровка
геометрии -> P2-E2 -> synthetic/real пары -> калибровка цвета -> P2-E7 ->
абляции. Числа smoke-прогона не являются результатами исследования.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import data as skdata

from .config import Config, load_config
from .datasets import hpatches as hp
from .datasets import mvtec as mv
from .experiments import (calibrate_color_from_runs, calibrate_hpatches, run_ablations, run_hpatches_geometry,
                          run_inspection, run_perturbations, summarize_inspection, write_summary)
from .io import write_png
from .synth import rotation_matrix
from .warp import warp_with_mask


def _base(name: str, size: tuple[int, int]) -> np.ndarray:
    img = getattr(skdata, name)()[..., :3].astype(np.float64) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_mini_hpatches(root: Path, rng: np.random.Generator) -> None:
    for seq, name in (("i_astro", "astronaut"), ("i_coffee", "coffee"), ("i_rocket", "rocket"),
                      ("v_cat", "chelsea"), ("v_rocket", "rocket"), ("v_astro", "astronaut")):
        d = root / seq
        d.mkdir(parents=True, exist_ok=True)
        E = _base(name, (480, 360))
        write_png(d / "1.png", E)
        for j in range(2, 7):
            if seq.startswith("i_"):
                H = np.eye(3)
                H[:2, 2] = rng.uniform(-2, 2, 2)
                gain = rng.uniform(0.6, 1.3)
            else:
                src = np.array([[0, 0], [479, 0], [479, 359], [0, 359]], float)
                dst = src + rng.uniform(-0.04, 0.04, (4, 2)) * np.array([480, 360]) * (j - 1) / 2
                H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(float)
                gain = 1.0
            I, _ = warp_with_mask(E, H, (480, 360))
            write_png(d / f"{j}.png", np.clip(I * gain, 0, 1))
            np.savetxt(d / f"H_1_{j}", H, fmt="%.10e")


def make_mini_mvtec(root: Path, rng: np.random.Generator) -> None:
    for cat, name in (("tex_coffee", "coffee"), ("tex_cat", "chelsea")):
        big = _base(name, (300, 300))

        def instance():
            M = rotation_matrix(rng.uniform(-1, 1), (300, 300))
            M[:2, 2] += rng.uniform(-2, 2, 2)
            img, _ = warp_with_mask(big, M, (300, 300))
            img = img[22:278, 22:278]
            return np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)

        for k in range(14):
            write_png(root / cat / "train" / "good" / f"{k:03d}.png", instance())
        for k in range(3):
            write_png(root / cat / "test" / "good" / f"{k:03d}.png", instance())
            img = instance()
            m = np.zeros((256, 256), bool)
            cy, cx = rng.integers(80, 176, 2)
            yy, xx = np.mgrid[0:256, 0:256]
            m[(yy - cy) ** 2 + (xx - cx) ** 2 <= 18 ** 2] = True
            img[m] = [0.9, 0.1, 0.1]
            write_png(root / cat / "test" / "blob" / f"{k:03d}.png", img)
            write_png(root / cat / "ground_truth" / "blob" / f"{k:03d}_mask.png", m)


def run_smoke(out: str | Path) -> None:
    out = Path(out)
    rng = np.random.default_rng(7)
    hroot, mroot = out / "data" / "hpatches", out / "data" / "mvtec"
    make_mini_hpatches(hroot, rng)
    make_mini_mvtec(mroot, rng)
    cfg = Config()

    man = hp.build_manifest(hroot)
    man.to_parquet(out / "hpatches_manifest.parquet", index=False)
    print("[direction check]", man["verdict"].value_counts().to_dict())

    e1 = run_hpatches_geometry(hroot, out / "P2-E1", cfg, "test", methods=("sift", "orb", "superpoint"),
                               manifest_path=out / "hpatches_manifest.parquet")
    print("[P2-E1]\n", e1.to_string())
    _, best = calibrate_hpatches(hroot, out / "geom_calib", cfg, "sift", subset="validation",
                                 manifest_path=out / "hpatches_manifest.parquet")
    print("[geometry calibration] best:", best)
    gcfg = load_config(out / "geom_calib" / "frozen_config.yaml") if best else cfg
    e2 = run_perturbations(hroot, out / "P2-E2", gcfg, "sift", subset="test", limit=1)
    print("[P2-E2]\n", e2[["factor", "value", "q_geom_rate", "success_3px", "mean_visibility"]].to_string())

    cats = ("tex_coffee", "tex_cat")
    syn = {sp: pd.concat([mv.build_synthetic(mroot, out / "synthetic", c, sp, variants_per_original=2)
                          for c in cats], ignore_index=True)
           for sp in ("synthetic_validation", "synthetic_test")}
    nval = pd.concat([mv.normal_validation_pairs(mroot, c) for c in cats], ignore_index=True)
    real = pd.concat([mv.real_test_pairs(mroot, c) for c in cats], ignore_index=True)
    val_pairs = pd.concat([syn["synthetic_validation"], nval], ignore_index=True)

    vpred, vreg = run_inspection(val_pairs, out / "inspect_validation", gcfg, label="validation", diagnostics=2)
    frozen, cbest = calibrate_color_from_runs(vpred, vreg, gcfg, out / "color_calib")
    print("[color calibration]", cbest)
    fcfg = frozen or gcfg
    for name, pairs in (("synthetic_test", syn["synthetic_test"]), ("real_test", real)):
        pred, _ = run_inspection(pairs, out / f"inspect_{name}", fcfg, label=name)
        tables = summarize_inspection(pred)
        write_summary(out / f"inspect_{name}", tables, "summary_P2-E7")
        print(f"[P2-E7 {name}]\n", tables["overall"].T.to_string())
    abl = run_ablations(syn["synthetic_validation"], syn["synthetic_test"], out / "ablations", fcfg)
    print("[P2-E3..E6]\n", abl[["experiment", "variant", "protocol", "status", "recall_all", "fpr_all",
                                 "coverage"]].to_string())
    (out / "SMOKE_DONE.json").write_text(json.dumps({"geometry_best": best, "color_best": cbest},
                                                    default=float, indent=2))
