"""Чтение изображений и экспорт результатов с привязкой к pair_id (раздел 9):
correspondences.parquet, homographies.npz, region_metrics.parquet,
predictions.parquet, reject_counts.csv, диагностические PNG."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .pipeline import PairResult


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def read_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 0


def write_png(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.dtype == bool:
        u8 = img.astype(np.uint8) * 255
    else:
        u8 = np.clip(np.rint(np.asarray(img) * 255.0), 0, 255).astype(np.uint8)
    if u8.ndim == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), u8):
        raise OSError(f"cannot write {path}")


def _opt(v):
    return None if v is None else float(v)


def prediction_record(r: PairResult) -> dict:
    g, e, a, d = r.geom, r.est, r.align, r.decision
    rec = {
        "pair_id": r.pair_id, "alignment": r.alignment, "method": r.method,
        "estimator": e.method if e else None,
        "n_kp_I": len(r.fI) if r.fI else None, "n_kp_E": len(r.fE) if r.fE else None,
        "n0": g.n0 if g else 0, "n_in": g.n_in if g else 0,
        "rho_in": _opt(g.rho_in) if g else None, "med_err": _opt(g.med_err) if g else None,
        "c_I": g.c_I if g else None, "c_E": g.c_E if g else None,
        "cond": g.adm.cond if g and g.adm else None,
        "adm_ok": bool(g.adm.ok) if g and g.adm else False,
        "ransac_status": e.status if e else None, "n_trials": e.n_trials if e else 0,
        "n_degenerate": e.n_degenerate if e else 0, "n_dlt_failed": e.n_dlt_failed if e else 0,
        "stop_reason": e.stop_reason if e else None, "required_trials": e.required_trials if e else None,
        "refine_status": e.refine_status if e else None,
        "q_geom": r.q_geom, "geom_reasons": ";".join(g.reasons) if g else "",
        "rho_vis": _opt(a.rho_vis) if a else None, "common_area": a.area if a else 0,
        "q_align": r.q_align, "align_reasons": ";".join(a.reasons) if a else "",
        "n_valid_regions": int(r.regions.kval.sum()) if r.regions is not None else 0,
        "compensation": json.dumps(r.regions.compensation) if r.regions is not None else None,
        "q_cmp": r.q_cmp, "D_glob": _opt(d.D) if d else None, "B_loc": _opt(d.B) if d else None,
        "crit_problem": d.crit_problem if d else None,
        "y_pred": d.y_pred if d else None,
        "violations": ";".join(d.violations) if d else "",
        "reject_reason": r.reject_reason() if r.y_pred is None else "",
    }
    rec.update({f"t_{k}": v for k, v in r.timings.items()})
    return rec


def correspondence_frame(r: PairResult) -> pd.DataFrame:
    if r.matches is None or len(r.matches) == 0:
        return pd.DataFrame()
    m = r.matches
    df = pd.DataFrame({
        "pair_id": r.pair_id, "i_I": m.pairs[:, 0], "j_E": m.pairs[:, 1],
        "x_I": r.x[:, 0], "y_I": r.x[:, 1], "u_E": r.u[:, 0], "v_E": r.u[:, 1],
        "d1": m.d1, "d2": m.d2, "ratio": m.ratio,
    })
    if r.est is not None and len(r.est.errors) == len(df):
        df["e_ij"] = r.est.errors
        df["inlier"] = r.est.inliers
    return df


def region_frame(r: PairResult) -> pd.DataFrame:
    g = r.regions
    if g is None:
        return pd.DataFrame()
    K = g.grid * g.grid
    df = pd.DataFrame({
        "pair_id": r.pair_id, "k": np.arange(K), "row": np.arange(K) // g.grid, "col": np.arange(K) % g.grid,
        "area_full": g.area_full, "area_avail": g.area_avail, "v": g.v, "in_kval": g.kval,
        "d_mean": g.d_mean, "d_of_means": g.d_of_means, "w": g.w,
        "critical": np.isin(np.arange(K), g.critical),
    })
    for t, row in zip(g.tau_pix, g.a):
        df[f"a_tau{t:g}"] = row
    if r.decision is not None and r.decision.b is not None:
        df["b"] = r.decision.b
    return df


def homography_arrays(r: PairResult) -> dict[str, np.ndarray]:
    out = {}
    if r.H is not None:
        out[f"{r.pair_id}__H"] = r.H
    if r.est is not None:
        if r.est.H0 is not None:
            out[f"{r.pair_id}__H0"] = r.est.H0
        out[f"{r.pair_id}__trials"] = r.est.trial_log
    if r.fI is not None:
        out[f"{r.pair_id}__A_I"] = r.fI.A
    if r.fE is not None:
        out[f"{r.pair_id}__A_E"] = r.fE.A
    return out


class Exporter:
    """Накопление результатов пар и запись файлов раздела 9."""

    def __init__(self, out_dir: str | Path, diagnostics: int = 0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.pred, self.corr, self.reg = [], [], []
        self.H: dict[str, np.ndarray] = {}
        self.diag_left = diagnostics

    def add(self, r: PairResult, extra: dict | None = None, E: np.ndarray | None = None) -> dict:
        rec = prediction_record(r)
        if extra:
            rec.update(extra)
        self.pred.append(rec)
        c = correspondence_frame(r)
        if len(c):
            self.corr.append(c)
        g = region_frame(r)
        if len(g):
            self.reg.append(g)
        self.H.update(homography_arrays(r))
        if self.diag_left > 0 and E is not None:
            self.diag_left -= 1
            save_diagnostics(self.out / "diagnostics", r, E)
        return rec

    def write(self) -> pd.DataFrame:
        pred = pd.DataFrame(self.pred)
        pred.to_parquet(self.out / "predictions.parquet", index=False)
        (pd.concat(self.corr, ignore_index=True) if self.corr else pd.DataFrame()).to_parquet(
            self.out / "correspondences.parquet", index=False)
        (pd.concat(self.reg, ignore_index=True) if self.reg else pd.DataFrame()).to_parquet(
            self.out / "region_metrics.parquet", index=False)
        np.savez_compressed(self.out / "homographies.npz", **self.H)
        reject_counts(pred).to_csv(self.out / "reject_counts.csv", index=False)
        return pred


def reject_counts(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame(columns=["stage", "reason", "count"])
    rows = []
    for stage, col in (("geometry", "geom_reasons"), ("alignment", "align_reasons"), ("final", "reject_reason")):
        if col not in pred:
            continue
        for s in pred[col].fillna(""):
            for reason in [x for x in s.split(";") if x and x != "Q_GEOM_0"]:
                rows.append((stage, reason))
    df = pd.DataFrame(rows, columns=["stage", "reason"])
    return df.value_counts().rename("count").reset_index() if len(df) else pd.DataFrame(
        columns=["stage", "reason", "count"])


def save_diagnostics(out_dir: Path, r: PairResult, E: np.ndarray) -> None:
    """Совмещённое изображение, маска Ω̂_com и карта δ00 (0…20 -> 0…1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if r.aligned is not None:
        write_png(out_dir / f"{r.pair_id}_aligned.png", r.aligned)
        write_png(out_dir / f"{r.pair_id}_valid.png", r.valid)
        overlay = 0.5 * E + 0.5 * r.aligned
        overlay[~r.valid] = E[~r.valid] * 0.3
        write_png(out_dir / f"{r.pair_id}_overlay.png", overlay)
    if r.regions is not None and r.regions.delta_map is not None:
        dm = np.nan_to_num(r.regions.delta_map, nan=0.0)
        heat = cv2.applyColorMap(np.clip(dm / 20.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(float) / 255
        heat[np.isnan(r.regions.delta_map)] = 0
        write_png(out_dir / f"{r.pair_id}_delta00.png", heat)
