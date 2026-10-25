"""Командная строка: python -m p2geom <команда> …"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config


def _csv(s: str | None) -> list[str] | None:
    return [x for x in s.split(",") if x] if s else None


def cmd_hpatches_manifest(a):
    from .datasets.hpatches import build_manifest
    df = build_manifest(a.root, verify=not a.no_verify)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    if "verdict" in df:
        print(df["verdict"].value_counts().to_string())


def cmd_hpatches_split(a):
    from .datasets.hpatches import default_split, list_sequences
    split = default_split([p.name for p in list_sequences(a.root)], a.val_fraction, a.seed)
    Path(a.out).write_text(json.dumps(split, indent=2))
    print({k: len(v) for k, v in split.items()})


def cmd_geometry(a):
    from .experiments import run_hpatches_geometry
    s = run_hpatches_geometry(a.root, a.out, load_config(a.config), a.subset, a.split, _csv(a.methods),
                              a.limit, a.manifest)
    print(s.to_string())


def cmd_calibrate_geometry(a):
    from .experiments import calibrate_hpatches
    _, best = calibrate_hpatches(a.root, a.out, load_config(a.config), a.method, a.split, a.subset, a.limit,
                                 a.manifest)
    print("best:", best)


def cmd_perturb(a):
    from .experiments import run_perturbations
    print(run_perturbations(a.root, a.out, load_config(a.config), a.method, a.split, a.subset, a.limit,
                            _csv(a.factors)).to_string())


def cmd_mvtec_build(a):
    from .datasets.mvtec import MAIN_CATEGORIES, build_synthetic, load_split, scan_category
    for cat in _csv(a.categories) or MAIN_CATEGORIES:
        train, _ = scan_category(a.root, cat)
        split = load_split(a.split, cat, train)
        for sp in _csv(a.splits):
            man = build_synthetic(a.root, a.out, cat, sp, split, a.variants, a.max_originals,
                                  min_defect_visible=None if a.hidden else 0.5)
            print(cat, sp, len(man), "pairs")


def cmd_mvtec_pairs(a):
    from .datasets.mvtec import MAIN_CATEGORIES, load_split, normal_validation_pairs, real_test_pairs, scan_category
    fn = {"real_test": real_test_pairs, "normal_validation": normal_validation_pairs}[a.kind]
    frames = []
    for cat in _csv(a.categories) or MAIN_CATEGORIES:
        train, _ = scan_category(a.root, cat)
        frames.append(fn(a.root, cat, load_split(a.split, cat, train)))
    df = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(len(df), "pairs ->", a.out)


def _read_pairs(paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def cmd_inspect(a):
    from .experiments import run_inspection, summarize_inspection, write_summary
    pred, _ = run_inspection(_read_pairs(a.pairs), a.out, load_config(a.config), alignment=a.alignment,
                             mask_mode=a.mask_mode, diagnostics=a.diagnostics, label="inspect")
    tables = summarize_inspection(pred)
    write_summary(Path(a.out), tables, "summary_P2-E7")
    print(tables["overall"].T.to_string())


def cmd_calibrate_color(a):
    from .experiments import calibrate_color_from_runs
    run = Path(a.run)
    cfg = load_config(a.config or run / "config.yaml")
    frozen, best = calibrate_color_from_runs(pd.read_parquet(run / "predictions.parquet"),
                                             pd.read_parquet(run / "region_metrics.parquet"), cfg, a.out)
    print("best:", best)


def cmd_ablations(a):
    from .experiments import run_ablations
    t = run_ablations(_read_pairs(a.val), _read_pairs(a.test), a.out, load_config(a.config), a.mask_mode,
                      not a.no_recalibrate)
    print(t.to_string())


def cmd_smoke(a):
    from .smoke import run_smoke
    run_smoke(a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="p2geom", description="P2 geometric (ТЗ 02)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    s = add("hpatches-manifest", cmd_hpatches_manifest, "проверка формата H и направления, manifest")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--no-verify", action="store_true")

    s = add("hpatches-split", cmd_hpatches_split, "временное разбиение (пока нет ТЗ 00)")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-fraction", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=2026)

    for name, fn, help_ in (("geometry", cmd_geometry, "P2-E1 на HPatches"),
                            ("calibrate-geometry", cmd_calibrate_geometry, "калибровка τr, n_min, c_min"),
                            ("perturb", cmd_perturb, "P2-E2: контролируемые воздействия")):
        s = add(name, fn, help_)
        s.add_argument("--root", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--config")
        s.add_argument("--split")
        s.add_argument("--subset", default="validation" if name == "calibrate-geometry" else "test")
        s.add_argument("--limit", type=int)
        if name == "geometry":
            s.add_argument("--methods", default="sift,orb,superpoint")
        else:
            s.add_argument("--method", default="sift")
        if name == "perturb":
            s.add_argument("--factors")
        else:
            s.add_argument("--manifest")

    s = add("mvtec-build", cmd_mvtec_build, "synthetic_validation / synthetic_test")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--categories")
    s.add_argument("--split")
    s.add_argument("--splits", default="synthetic_validation,synthetic_test")
    s.add_argument("--variants", type=int, default=2)
    s.add_argument("--max-originals", type=int)
    s.add_argument("--hidden", action="store_true", help="опыт скрытых дефектов: без ограничения видимости")

    s = add("mvtec-pairs", cmd_mvtec_pairs, "пары real_test или normal_validation")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--kind", choices=["real_test", "normal_validation"], required=True)
    s.add_argument("--categories")
    s.add_argument("--split")

    s = add("inspect", cmd_inspect, "конвейер inspection + сводка P2-E7")
    s.add_argument("--pairs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--alignment", choices=["estimated", "oracle", "none"], default="estimated")
    s.add_argument("--mask-mode", choices=["unknown", "known"], default="unknown")
    s.add_argument("--diagnostics", type=int, default=0, help="число пар с диагностическими PNG")

    s = add("calibrate-color", cmd_calibrate_color, "цветовые пороги по прогону validation")
    s.add_argument("--run", required=True, help="каталог прогона inspect на validation")
    s.add_argument("--out", required=True)
    s.add_argument("--config")

    s = add("ablations", cmd_ablations, "P2-E3…P2-E6")
    s.add_argument("--val", nargs="+", required=True)
    s.add_argument("--test", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--mask-mode", choices=["unknown", "known"], default="unknown")
    s.add_argument("--no-recalibrate", action="store_true")

    s = add("smoke", cmd_smoke, "сквозной прогон на встроенных изображениях skimage")
    s.add_argument("--out", default="runs/smoke")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
