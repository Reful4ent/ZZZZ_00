"""Раннеры опытов P2-E1…P2-E7, калибровки и сводки (раздел 8).

Тяжёлые прогоны (признаки, RANSAC, warp, карты ΔE) выполняются один раз на
вариант совмещения; варианты gate, правила решения, критических участков и
агрегации участка пересчитываются по сохранённым метрикам (calibration.recompute)
теми же формулами, что и в конвейере.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (GEOM_RATIOS, calibrate_color, calibrate_geometry, freeze_color,
                          freeze_geometry, gate_variant, recompute, stack_regions)
from .config import Config, save_config
from .datasets import hpatches as hp
from .datasets.mvtec import aggregate_multi_reference
from .features import detect_describe
from .io import Exporter, read_mask, read_rgb
from .metrics import homography_error, inspection_metrics, p_corr, repeatability, success_at
from .pipeline import PairResult, image_size, run_geometry, run_pair
from .regions import critical_cells
from .synth import SYNTH_AREA_RATIO_RANGE, factor_cases, generate_case


class LRU:
    def __init__(self, fn, size: int = 32):
        self.fn, self.size, self.d = fn, size, OrderedDict()

    def __call__(self, key, *args):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = self.fn(key, *args)
        self.d[key] = v
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return v


def _feature_cache(cfg: Config) -> LRU:
    return LRU(lambda key, img: detect_describe(img, cfg.features), 64)


def _write_meta(out: Path, cfg: Config, **extra) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    (out / "run_meta.json").write_text(json.dumps({"config_fingerprint": cfg.fingerprint(), **extra},
                                                  ensure_ascii=False, indent=2, default=str))


# ---------------------------------------------------------------- geometry (HPatches)


def geometry_eval(res: PairResult, H_gt: np.ndarray | None, size_I, size_E) -> dict:
    """Метрики оценщика для одной пары (Г-3, Г-4, Г-5)."""
    out = {"P_corr": None, "n_corr": None, "Rep": None, "n_rep": None, "n_I_vis": None, "n_E_vis": None,
           "e_H": None}
    if H_gt is None:
        return out
    out["P_corr"], out["n_corr"] = p_corr(res.x, res.u, H_gt)
    if res.fI is not None and res.fE is not None and res.fI.status == "OK":
        out["Rep"], out["n_rep"], out["n_I_vis"], out["n_E_vis"] = repeatability(
            res.fI.kp_orig, res.fE.kp_orig, H_gt, size_I, size_E)
    adm_ok = bool(res.geom and res.geom.adm and res.geom.adm.ok)
    if res.H is not None and adm_ok:
        out["e_H"] = homography_error(res.H, H_gt, size_I, size_E)
    return out


def _hpatches_pairs(root, split_path, subset, limit, manifest_path, negatives=True):
    seqs = [p.name for p in hp.list_sequences(root)]
    split = hp.load_split(split_path, seqs)
    names = split[subset][:limit] if limit else split[subset]
    manifest = pd.read_parquet(manifest_path) if manifest_path else None
    return hp.make_pairs(root, names, manifest, negatives=negatives), split


def run_hpatches_geometry(root, out, cfg: Config, subset: str = "test", split_path=None,
                          methods=("sift", "orb", "superpoint"), limit=None, manifest_path=None) -> pd.DataFrame:
    """P2-E1: точки, пары, P_corr, Rep, e_H, доля успехов, Coverage, время;
    i_ и v_ раздельно; отрицательные пары — отдельно."""
    out = Path(out)
    pairs, split = _hpatches_pairs(root, split_path, subset, limit, manifest_path)
    imgs = LRU(lambda p: read_rgb(p), 24)
    summaries = []
    for method in methods:
        mcfg = cfg.replace(features={"method": method})
        mout = out / method
        _write_meta(mout, mcfg, experiment="P2-E1", subset=subset, n_pairs=len(pairs))
        feats = _feature_cache(mcfg)
        exp = Exporter(mout)
        not_run = None
        for p in pairs:
            E, I = imgs(p.ref_path), imgs(p.query_path)
            fE, fI = feats(p.ref_path, E), feats(p.query_path, I)
            if fE.status != "OK":
                not_run = fE.info
                break
            res = run_geometry(E, I, mcfg, p.pair_id, fI, fE)
            ev = geometry_eval(res, p.H_gt, image_size(I), image_size(E))
            exp.add(res, {"part": p.part, "seq": p.seq, "ref_seq": p.ref_seq, "j": p.j,
                          "negative": p.negative, **ev})
        if not_run is not None:
            (mout / "NOT_RUN.json").write_text(json.dumps(not_run, ensure_ascii=False, indent=2))
            summaries.append(pd.DataFrame([{"method": method, "status": "NOT_RUN",
                                            "reason": not_run.get("reason")}]))
            continue
        pred = exp.write()
        s = summarize_geometry(pred)
        s.insert(0, "method", method)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary_P2-E1.csv", index=False)
    return summary


def summarize_geometry(pred: pd.DataFrame, by=("part",)) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(list(by) + ["negative"], dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        neg = bool(key[-1])
        r = dict(zip(list(by) + ["negative"], key))
        r.update(status="OK", n_pairs=len(g), q_geom_rate=float(g["q_geom"].mean()),
                 mean_kp=float(pd.concat([g["n_kp_I"], g["n_kp_E"]]).mean()),
                 mean_M0=float(g["n0"].mean()), empty_M0_frac=float((g["n0"] == 0).mean()),
                 mean_time_s=float(g.filter(regex="^t_").sum(axis=1).mean()))
        if neg:
            r["false_accept_rate"] = r["q_geom_rate"]
        else:
            eH = g["e_H"].tolist()
            succ = success_at(eH)
            ok3 = (g["q_geom"] == 1) & (g["e_H"].astype(float).fillna(np.inf) <= 3.0)
            r.update(mean_P_corr=float(g["P_corr"].astype(float).mean()),
                     total_n_corr=float(g["n_corr"].astype(float).sum()),
                     mean_Rep=float(g["Rep"].astype(float).mean()),
                     median_e_H=float(g["e_H"].astype(float).median()),
                     success_1px=succ[1.0], success_3px=succ[3.0], success_5px=succ[5.0],
                     q_geom_and_eH_le3=float(ok3.mean()),
                     mean_c_I=float(g["c_I"].astype(float).mean()), mean_c_E=float(g["c_E"].astype(float).mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def calibrate_hpatches(root, out, cfg: Config, method: str = "sift", split_path=None, subset="validation",
                       limit=None, manifest_path=None) -> tuple[pd.DataFrame, dict | None]:
    """Геометрическая калибровка τr × n_min × c_min на validation; результат замораживается."""
    out = Path(out)
    pairs, _ = _hpatches_pairs(root, split_path, subset, limit, manifest_path)
    mcfg = cfg.replace(features={"method": method})
    _write_meta(out, mcfg, experiment="geometry_calibration", subset=subset)
    imgs = LRU(lambda p: read_rgb(p), 24)
    feats = _feature_cache(mcfg)
    stats = []
    for tau in GEOM_RATIOS:
        tcfg = mcfg.replace(matching={"ratio": tau})
        for p in pairs:
            E, I = imgs(p.ref_path), imgs(p.query_path)
            res = run_geometry(E, I, tcfg, p.pair_id, feats(p.query_path, I), feats(p.ref_path, E))
            g = res.geom
            ev = geometry_eval(res, p.H_gt, image_size(I), image_size(E)) if not p.negative else {"e_H": None}
            stats.append({"ratio": tau, "pair_id": p.pair_id, "part": p.part, "negative": p.negative,
                          "n_in": g.n_in, "rho_in": g.rho_in, "med_err": g.med_err, "c_I": g.c_I, "c_E": g.c_E,
                          "adm_ok": bool(g.adm and g.adm.ok), "e_H": ev["e_H"]})
    sdf = pd.DataFrame(stats)
    sdf.to_parquet(out / "calibration_stats.parquet", index=False)
    table, best = calibrate_geometry(sdf, mcfg.gate)
    table.to_csv(out / "geometry_calibration_table.csv", index=False)
    (out / "geometry_calibration_best.json").write_text(json.dumps(best, indent=2, default=float))
    if best is not None:
        save_config(freeze_geometry(mcfg, best), out / "frozen_config.yaml")
    return table, best


def run_perturbations(root, out, cfg: Config, method: str = "sift", split_path=None, subset="test",
                      limit=None, factors=None) -> pd.DataFrame:
    """P2-E2: однофакторные воздействия на image1 последовательностей HPatches."""
    out = Path(out)
    seqs = [p.name for p in hp.list_sequences(root)]
    split = hp.load_split(split_path, seqs)
    names = split[subset][:limit] if limit else split[subset]
    scfg = cfg.replace(features={"method": method}, gate={"area_ratio_range": SYNTH_AREA_RATIO_RANGE})
    _write_meta(out, scfg, experiment="P2-E2", subset=subset)
    cases = [c for c in factor_cases() if factors is None or c["factor"] in factors]
    exp = Exporter(out)
    for name in names:
        E = read_rgb(hp._image_path(Path(root) / name, 1))
        fE = detect_describe(E, scfg.features)
        for case in cases:
            obs = generate_case(E, case)
            pid = f"{name}__{case['case_id']}"
            res = run_geometry(E, obs.I, scfg, pid, fE=fE)
            ev = geometry_eval(res, obs.H_gt, image_size(obs.I), image_size(E))
            exp.add(res, {"seq": name, "part": name[0], "negative": False, "factor": case["factor"],
                          "factor_value": str(case["value"]), "case_seed": case["seed"],
                          "gen_attempts": obs.meta["attempts"], "visibility": obs.visibility,
                          "G": json.dumps(obs.G.tolist()), **ev})
    pred = exp.write()
    curves = []
    for (f, v), g in pred.groupby(["factor", "factor_value"], sort=False):
        succ = success_at(g["e_H"].tolist())
        reasons = g["geom_reasons"].str.split(";").explode()
        reasons = reasons[reasons != ""].value_counts()
        curves.append({"factor": f, "value": v, "n": len(g), "q_geom_rate": float(g["q_geom"].mean()),
                       "success_1px": succ[1.0], "success_3px": succ[3.0], "success_5px": succ[5.0],
                       "median_e_H": float(g["e_H"].astype(float).median()),
                       "mean_visibility": float(g["visibility"].mean()),
                       "reject_reasons": json.dumps(reasons.to_dict())})
    cdf = pd.DataFrame(curves)
    cdf.to_csv(out / "curves_P2-E2.csv", index=False)
    return cdf


# ---------------------------------------------------------------- inspection (MVTec)


def run_inspection(pairs: pd.DataFrame, out, cfg: Config, *, alignment: str = "estimated",
                   mask_mode: str = "unknown", diagnostics: int = 0, label: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Тяжёлый прогон конвейера по таблице пар (E_path, I_path, y_match, …).
    mask_mode: known — маска видимости I передаётся (ГЕОМ-04), unknown — нет.
    Результаты known/unknown пишутся в разные каталоги и не объединяются."""
    if mask_mode not in ("known", "unknown"):
        raise ValueError(mask_mode)
    out = Path(out)
    _write_meta(out, cfg, experiment=label, alignment=alignment, mask_mode=mask_mode, n_pairs=len(pairs))
    imgs = LRU(lambda p: read_rgb(p), 16)
    feats = _feature_cache(cfg)
    exp = Exporter(out, diagnostics)
    meta_cols = [c for c in pairs.columns if c not in ("E_path", "I_path", "visible_path", "change_mask_path",
                                                       "mask_path", "G", "H_gt")]
    for row in pairs.to_dict("records"):
        E, I = imgs(row["E_path"]), imgs(row["I_path"])
        vis = read_mask(row["visible_path"]) if mask_mode == "known" and row.get("visible_path") else None
        H_or = np.array(json.loads(row["H_gt"])) if alignment == "oracle" else None
        fI = fE = None
        if alignment == "estimated":
            fI, fE = feats(row["I_path"], I), feats(row["E_path"], E)
        res = run_pair(E, I, cfg, row["pair_id"], vis_mask_I=vis, alignment=alignment, H_oracle=H_or,
                       keep_map=diagnostics > 0, fI=fI, fE=fE)
        exp.add(res, {c: row[c] for c in meta_cols if c != "pair_id"} | {"mask_mode": mask_mode}, E=E)
    pred = exp.write()
    reg = pd.read_parquet(out / "region_metrics.parquet")
    return pred, reg


def final_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """Строки, по которым принимается итоговое решение: synthetic — каждая пара;
    real/normal validation (есть query_id) — выбранный эталон запроса."""
    if "query_id" in pred and pred["query_id"].notna().any():
        multi = pred[pred["query_id"].notna()]
        single = pred[pred["query_id"].isna()]
        return pd.concat([single, aggregate_multi_reference(multi)], ignore_index=True)
    return pred


def summarize_inspection(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """P2-E7: матрица 2×3, ложные принятия/отклонения, отказы по категориям и площадям."""
    if "mask_mode" in pred and pred["mask_mode"].nunique() > 1:
        raise ValueError("known-mask and unknown-mask results must never be combined")
    fr = final_rows(pred)
    y_pred = [None if pd.isna(v) else int(v) for v in fr["y_pred"]]
    overall = pd.DataFrame([inspection_metrics(fr["y_match"].to_numpy(int), y_pred)])
    by_cat = []
    for cat, g in fr.groupby("category"):
        m = inspection_metrics(g["y_match"].to_numpy(int), [None if pd.isna(v) else int(v) for v in g["y_pred"]])
        by_cat.append({"category": cat, **m})
    out = {"overall": overall, "by_category": pd.DataFrame(by_cat)}
    neg = fr[fr["y_match"] == 0]
    if "defect_area" in neg and neg["defect_area"].notna().any():
        grp = neg.groupby(["category", "defect_type", "defect_area"], dropna=False)
        out["by_defect"] = grp.apply(lambda g: pd.Series({
            "n": len(g), "detected": float((g["y_pred"] == 0).mean()),
            "missed": float((g["y_pred"] == 1).mean()), "rejected": float(g["y_pred"].isna().mean())}),
            include_groups=False).reset_index()
    if "defect_visible_frac" in neg and neg["defect_visible_frac"].notna().any():
        # Опыт скрытых дефектов: полностью скрытые случаи выводятся отдельно.
        bins = pd.cut(neg["defect_visible_frac"].astype(float), [-0.01, 0.0, 0.5 - 1e-12, 1.0],
                      labels=["fully_hidden", "visible_lt_0.5", "visible_ge_0.5"])
        out["by_defect_visibility"] = neg.groupby(bins, observed=False).apply(lambda g: pd.Series({
            "n": len(g), "detected": float((g["y_pred"] == 0).mean()) if len(g) else float("nan"),
            "rejected": float(g["y_pred"].isna().mean()) if len(g) else float("nan")}),
            include_groups=False).reset_index()
    rej = fr[fr["y_pred"].isna()]
    out["rejects"] = (rej.groupby(["category", "reject_reason"]).size().rename("count").reset_index()
                      if len(rej) and "reject_reason" in rej else pd.DataFrame())
    return out


def write_summary(out: Path, tables: dict[str, pd.DataFrame], prefix: str) -> None:
    for name, df in tables.items():
        df.to_csv(out / f"{prefix}_{name}.csv", index=False)


def calibrate_color_from_runs(pred: pd.DataFrame, reg: pd.DataFrame, cfg: Config, out=None) -> tuple[Config | None, dict]:
    """Цветовая калибровка по прогонам validation (synthetic validation + normal
    validation против dev-галереи). Геометрические пороги уже заморожены в cfg."""
    pred = recompute(pred, reg, cfg)
    fr = final_rows(pred)
    st = stack_regions(fr["pair_id"].tolist(), reg, cfg.compare.grid)
    crit = critical_cells(cfg.compare.grid, cfg.compare.critical)
    table, best = calibrate_color(fr["y_match"].to_numpy(int), fr["q_cmp"].to_numpy(bool), st, crit,
                                  cfg.compare.region_stat, cfg.decision)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out / "color_calibration_table.parquet", index=False)
        (out / "color_calibration_best.json").write_text(json.dumps(best, indent=2, default=float))
    if best["status"] != "ok":
        return None, best
    frozen = freeze_color(cfg, best)
    if out is not None:
        save_config(frozen, Path(out) / "frozen_config.yaml")
    return frozen, best


# ---------------------------------------------------------------- абляции P2-E3…P2-E6


def decision_variants(cfg: Config) -> dict[str, tuple[str, Config]]:
    """Варианты, пересчитываемые без повторного warp: (эксперимент, cfg)."""
    v = {}
    for mode in ("none", "support", "coverage", "full"):
        v[f"gate_{mode}"] = ("P2-E4", gate_variant(cfg, mode))
    v["stat_mean_delta"] = ("P2-E5", cfg.replace(compare={"region_stat": "mean_delta"}))
    v["stat_delta_of_means"] = ("P2-E5", cfg.replace(compare={"region_stat": "delta_of_means"}))
    v["rule_D"] = ("P2-E6", cfg.replace(decision={"use_B": False, "use_critical": False},
                                        compare={"critical": "none"}))
    v["rule_D_B"] = ("P2-E6", cfg.replace(decision={"use_B": True, "use_critical": False},
                                          compare={"critical": "none"}))
    v["rule_D_B_critical"] = ("P2-E6", cfg.replace(decision={"use_B": True, "use_critical": True},
                                                   compare={"critical": "center4"}))
    return v


def run_ablations(val_pairs: pd.DataFrame, test_pairs: pd.DataFrame, out, cfg: Config,
                  mask_mode: str = "unknown", recalibrate: bool = True) -> pd.DataFrame:
    """Все варианты в двух протоколах: fixed (общие параметры cfg) и, при
    recalibrate, отдельно перекалиброванные с одинаковым validation-бюджетом.
    Основной результат — fixed-протокол исходного cfg; лучший вариант постфактум
    не выбирается."""
    out = Path(out)
    heavy = {"estimated": ("estimated", cfg), "oracle": ("oracle", cfg), "none": ("none", cfg),
             "compensation_on": ("estimated", cfg.replace(compare={"compensation": True}))}
    has_gt = "H_gt" in test_pairs and test_pairs["H_gt"].notna().all() and \
        "H_gt" in val_pairs and val_pairs["H_gt"].notna().all()
    runs = {}
    for name, (align, hcfg) in heavy.items():
        if align == "oracle" and not has_gt:
            continue
        runs[name] = {split: run_inspection(pairs, out / f"run_{name}" / split, hcfg, alignment=align,
                                            mask_mode=mask_mode, label=f"ablation:{name}")
                      for split, pairs in (("validation", val_pairs), ("test", test_pairs))}
    variants: list[tuple[str, str, str, Config]] = [
        ("P2-E3", "no_alignment", "none", cfg), ("P2-E3", "H_hat", "estimated", cfg),
        ("P2-E6", "compensation_off", "estimated", cfg),
        ("P2-E6", "compensation_on", "compensation_on", heavy["compensation_on"][1]),
    ]
    if "oracle" in runs:
        variants.append(("P2-E3", "H_gt_oracle", "oracle", cfg))
    for vname, (expname, vcfg) in decision_variants(cfg).items():
        variants.append((expname, vname, "estimated", vcfg))
    rows = []
    for expname, vname, run, vcfg in variants:
        (vp, vr), (tp, tr) = runs[run]["validation"], runs[run]["test"]
        protocols = [("fixed", vcfg)]
        if recalibrate:
            rcfg, best = calibrate_color_from_runs(vp, vr, vcfg)
            protocols.append(("recalibrated", rcfg))
        for proto, pcfg in protocols:
            if pcfg is None:
                rows.append({"experiment": expname, "variant": vname, "protocol": proto, "status": "infeasible"})
                continue
            res = final_rows(recompute(tp, tr, pcfg))
            m = inspection_metrics(res["y_match"].to_numpy(int),
                                   [None if pd.isna(v) else int(v) for v in res["y_pred"]])
            rows.append({"experiment": expname, "variant": vname, "protocol": proto, "status": "ok",
                         "primary": proto == "fixed" and vcfg == cfg and run == "estimated", **m})
    table = pd.DataFrame(rows)
    table.to_csv(out / f"ablations_{mask_mode}.csv", index=False)
    return table
