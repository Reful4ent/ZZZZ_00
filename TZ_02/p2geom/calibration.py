"""Калибровка на validation и пересчёт решений по сохранённым метрикам.

Геометрия (раздел 2): перебор τr × n_min × c_min, остальные пороги фиксированы.
Максимум доли положительных пар с q_geom=1 и e_H <= 3 px при доле q_geom=1 на
отрицательных <= 0.05; tie-break: меньше ложных принятий, затем строже n_min и
c_min, затем меньший τr.

Цвет (раздел 7): τpix × τd × τa × τB × τD. Максимум Recall_all при
FPR_all <= 0.05 и Coverage >= 0.5; tie-break: меньше FPR, больше Coverage, затем
лексикографически более строгие (меньшие) пороги. Недостижимо -> infeasible.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DecisionConfig, GeometryGateConfig
from .decision import aggregate, decide_arrays
from .regions import critical_cells
from .validate import gate_failures
from .warp import align_failures

GEOM_RATIOS = (0.65, 0.75, 0.85)
GEOM_N_MIN = (4, 8, 12, 20)
GEOM_C_MIN = (0.0, 0.05, 0.10, 0.20)
COLOR_TAU_PIX = (2.0, 5.0, 10.0)
COLOR_TAU_D = (2.0, 5.0, 10.0)
COLOR_TAU_A = (0.05, 0.10, 0.20)
COLOR_TAU_B = (0.05, 0.10, 0.20)


# ---------------------------------------------------------------- геометрия


def q_geom_from_stats(df: pd.DataFrame, gate: GeometryGateConfig) -> np.ndarray:
    fails = gate_failures(df["n_in"].to_numpy(), df["rho_in"].astype(float).to_numpy(),
                          df["med_err"].astype(float).to_numpy(), df["c_I"].astype(float).to_numpy(),
                          df["c_E"].astype(float).to_numpy(), gate)
    ok = np.array(df["adm_ok"].fillna(False).to_numpy(bool), copy=True)
    for bad in fails.values():
        ok &= ~np.asarray(bad, bool)
    return ok


def calibrate_geometry(stats: pd.DataFrame, base: GeometryGateConfig, e_thr: float = 3.0,
                       max_neg_accept: float = 0.05) -> tuple[pd.DataFrame, dict | None]:
    """stats: строки (ratio, pair) с колонками ratio, negative, n_in, rho_in, med_err,
    c_I, c_E, adm_ok, e_H."""
    rows = []
    for tau, g in stats.groupby("ratio"):
        neg = g["negative"].to_numpy(bool)
        pos = ~neg
        eH = g["e_H"].astype(float).fillna(np.inf).to_numpy()
        for n_min, c_min in itertools.product(GEOM_N_MIN, GEOM_C_MIN):
            gate = dataclasses.replace(base, n_min=n_min, c_I_min=c_min, c_E_min=c_min)
            q = q_geom_from_stats(g, gate)
            rows.append({"ratio": tau, "n_min": n_min, "c_min": c_min,
                         "pos_success": float((q & pos & (eH <= e_thr)).sum() / max(pos.sum(), 1)),
                         "neg_accept": float((q & neg).sum() / neg.sum()) if neg.any() else 0.0,
                         "pos_q_geom": float((q & pos).sum() / max(pos.sum(), 1)),
                         "n_pos": int(pos.sum()), "n_neg": int(neg.sum())})
    table = pd.DataFrame(rows)
    feas = table[table["neg_accept"] <= max_neg_accept]
    if feas.empty:
        return table, None
    best = feas.sort_values(["pos_success", "neg_accept", "n_min", "c_min", "ratio"],
                            ascending=[False, True, False, False, True], kind="stable").iloc[0]
    return table, best.to_dict()


def freeze_geometry(cfg: Config, best: dict) -> Config:
    return cfg.replace(matching={"ratio": float(best["ratio"])},
                       gate={"n_min": int(best["n_min"]), "c_I_min": float(best["c_min"]),
                             "c_E_min": float(best["c_min"])})


# ---------------------------------------------------------------- участки из таблиц


@dataclass
class RegionStack:
    pair_ids: list[str]
    kval: np.ndarray          # (n, K)
    d_mean: np.ndarray
    d_of_means: np.ndarray
    w: np.ndarray
    a: dict[float, np.ndarray]

    def d(self, stat: str) -> np.ndarray:
        return self.d_mean if stat == "mean_delta" else self.d_of_means


def stack_regions(pair_ids: list[str], reg: pd.DataFrame, grid: int) -> RegionStack:
    n, K = len(pair_ids), grid * grid
    pos = {p: i for i, p in enumerate(pair_ids)}
    kval = np.zeros((n, K), bool)
    dm, dom, w = np.full((n, K), np.nan), np.full((n, K), np.nan), np.zeros((n, K))
    a_cols = [c for c in reg.columns if c.startswith("a_tau")] if len(reg) else []
    a = {float(c[5:]): np.full((n, K), np.nan) for c in a_cols}
    if len(reg):
        r = reg[reg["pair_id"].isin(pos)]
        if len(r) and r["k"].max() >= K:
            raise ValueError(f"region table has k >= {K}: grid mismatch")
        i = r["pair_id"].map(pos).to_numpy()
        k = r["k"].to_numpy()
        kval[i, k] = r["in_kval"].to_numpy(bool)
        dm[i, k] = r["d_mean"].to_numpy(float)
        dom[i, k] = r["d_of_means"].to_numpy(float)
        w[i, k] = r["w"].to_numpy(float)
        for c in a_cols:
            a[float(c[5:])][i, k] = r[c].to_numpy(float)
    return RegionStack(list(pair_ids), kval, dm, dom, w, a)


def recompute(pred: pd.DataFrame, reg: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Пересчёт q_geom, q_align, q_cmp, ŷ по сохранённым метрикам для вариантов,
    не меняющих совмещение и карты (gate, правило решения, критические участки,
    способ агрегации участка)."""
    out = pred.copy()
    est = out["alignment"].to_numpy() == "estimated"
    qg = np.array(out["q_geom"].to_numpy(bool), copy=True)
    if est.any():
        qg[est] = q_geom_from_stats(out[est], cfg.gate)
    qa = qg.copy()
    for bad in align_failures(out["rho_vis"].astype(float).to_numpy(), out["common_area"].to_numpy(),
                              cfg.align).values():
        qa &= ~np.asarray(bad, bool)
    st = stack_regions(out["pair_id"].tolist(), reg, cfg.compare.grid)
    crit = critical_cells(cfg.compare.grid, cfg.compare.critical)
    r = decide_arrays(qa, st.kval, st.d(cfg.compare.region_stat), st.a[cfg.decision.tau_pix], st.w, crit,
                      cfg.decision)
    out["q_geom"], out["q_align"], out["q_cmp"] = qg.astype(int), qa.astype(int), r["q_cmp"].astype(int)
    out["D_glob"], out["B_loc"] = r["D"], r["B"]
    out["y_pred"] = r["y_pred"]
    return out


# ---------------------------------------------------------------- цвет


def calibrate_color(y_true: np.ndarray, q_cmp: np.ndarray, st: RegionStack, critical: list[int],
                    stat: str, base: DecisionConfig, fpr_max: float = 0.05,
                    cov_min: float = 0.5) -> tuple[pd.DataFrame, dict]:
    y_true = np.asarray(y_true, int)
    q = np.asarray(q_cmp, bool)
    n0, n1 = int((y_true == 0).sum()), int((y_true == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("color calibration needs both y_match=0 and y_match=1 pairs")
    coverage = float(q.mean())
    d = st.d(stat)
    D, _, _ = aggregate(d, np.zeros_like(d), st.w, st.kval, np.inf, np.inf)   # D не зависит от τd, τa, τpix
    tD = np.concatenate([[-np.inf], np.unique(D[q]), [np.inf]])
    tau_B_grid = COLOR_TAU_B if base.use_B else (np.inf,)
    rows = []
    for tp, td, ta in itertools.product(COLOR_TAU_PIX, COLOR_TAU_D, COLOR_TAU_A):
        _, B, b = aggregate(d, st.a[tp], st.w, st.kval, td, ta)
        crit_bad = b[:, critical].any(1) if (critical and base.use_critical) else np.zeros(len(q), bool)
        for tb in tau_B_grid:
            hard = q & ((B > tb) | crit_bad)                 # ŷ=0 независимо от τD
            soft = q & ~hard

            def detected(cls):                               # число ŷ=0 при каждом пороге τD
                Ds = np.sort(D[soft & (y_true == cls)])
                return (hard & (y_true == cls)).sum() + len(Ds) - np.searchsorted(Ds, tD, side="right")

            rec, fpr = detected(0) / n0, detected(1) / n1
            for t, rc, fp in zip(tD, rec, fpr):
                rows.append((tp, td, ta, tb, t, rc, fp))
    table = pd.DataFrame(rows, columns=["tau_pix", "tau_d", "tau_a", "tau_B", "tau_D", "recall_all", "fpr_all"])
    table["coverage"] = coverage
    feas = table[(table["fpr_all"] <= fpr_max) & (table["coverage"] >= cov_min)]
    if feas.empty:
        return table, {"status": "infeasible", "coverage": coverage}
    best = feas.sort_values(["recall_all", "fpr_all", "coverage", "tau_pix", "tau_d", "tau_a", "tau_B", "tau_D"],
                            ascending=[False, True, False, True, True, True, True, True], kind="stable").iloc[0]
    out = {"status": "ok", **{k: float(v) for k, v in best.items()}}
    return table, out


def freeze_color(cfg: Config, best: dict) -> Config:
    if best.get("status") != "ok":
        raise ValueError("color calibration infeasible: thresholds are not frozen")
    return cfg.replace(decision={"tau_pix": best["tau_pix"], "tau_d": best["tau_d"], "tau_a": best["tau_a"],
                                 "tau_B": best["tau_B"], "tau_D": best["tau_D"]})


def gate_variant(cfg: Config, mode: str) -> Config:
    """P2-E4: none | support | coverage | full. «none» оставляет только техническую
    допустимость H (конечность, обратимость, знаменатели) — без неё warp невозможен."""
    flags = {"none": (False, False), "support": (True, False), "coverage": (False, True), "full": (True, True)}
    s, c = flags[mode]
    return cfg.replace(gate={"use_support": s, "use_coverage": c})

