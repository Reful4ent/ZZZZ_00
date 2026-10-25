"""decide: q_cmp (Г2-122) и ŷ ∈ {1, 0, ⊥} (Г2-120, Г2-121, Г2-123).

ŷ = None (в массивах — NaN) означает отказ ⊥. ŷ = 0 — обнаружено
несоответствие, а не ошибка совмещения. decide_arrays — общее векторное ядро
для одиночного решения, абляций и пересчёта по сохранённым метрикам.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CompareConfig, DecisionConfig
from .regions import RegionMetrics


def aggregate(d: np.ndarray, a: np.ndarray, w: np.ndarray, kval: np.ndarray,
              tau_d: float, tau_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D_glob, B_loc и b_k по последней оси (участки), батчи (..., K).
    b_k = 1[d_k > τd ∨ a_k > τa] только для k ∈ K_val; вне K_val вес 0."""
    b = kval & ((np.nan_to_num(d, nan=-np.inf) > tau_d) | (np.nan_to_num(a, nan=-np.inf) > tau_a))
    D = np.sum(np.where(kval, w * np.nan_to_num(d), 0.0), axis=-1)
    B = np.sum(np.where(kval, w * b, 0.0), axis=-1)
    return D, B, b


def decide_arrays(q_align: np.ndarray, kval: np.ndarray, d: np.ndarray, a: np.ndarray, w: np.ndarray,
                  critical: list[int], dcfg: DecisionConfig) -> dict[str, np.ndarray]:
    """Батч из n пар, K участков: q_align (n,), остальные (n, K)."""
    kval = np.asarray(kval, bool)
    any_val = kval.any(axis=-1)
    req_ok = kval[:, critical].all(axis=-1) if critical else np.ones(len(kval), bool)
    q_cmp = np.asarray(q_align, bool) & any_val & req_ok
    D, B, b = aggregate(d, a, w, kval, dcfg.tau_d, dcfg.tau_a)
    crit_bad = b[:, critical].any(axis=-1) if critical else np.zeros(len(kval), bool)
    v_D = D > dcfg.tau_D
    v_B = (B > dcfg.tau_B) if dcfg.use_B else np.zeros_like(v_D)
    v_C = crit_bad if dcfg.use_critical else np.zeros_like(v_D)
    y = np.where(q_cmp, (~(v_D | v_B | v_C)).astype(float), np.nan)
    return {"q_cmp": q_cmp, "any_val": any_val, "req_ok": req_ok, "D": np.where(any_val, D, np.nan),
            "B": np.where(any_val, B, np.nan), "b": b, "crit_bad": crit_bad,
            "v_D": v_D, "v_B": v_B, "v_C": v_C, "y_pred": y}


@dataclass
class Decision:
    q_cmp: int
    y_pred: int | None
    D: float | None = None
    B: float | None = None
    b: np.ndarray | None = None
    crit_problem: bool | None = None
    reasons: list[str] = field(default_factory=list)      # причины отказа (q_cmp = 0)
    violations: list[str] = field(default_factory=list)   # нарушенные условия соответствия (ŷ = 0)


def decide(q_align: int, reg: RegionMetrics | None, dcfg: DecisionConfig, ccfg: CompareConfig) -> Decision:
    if reg is None:
        return Decision(0, None, reasons=(["Q_ALIGN_0"] if not q_align else []) + ["NO_REGIONS"])
    r = decide_arrays(np.array([q_align]), reg.kval[None], reg.d(ccfg.region_stat)[None],
                      reg.a_at(dcfg.tau_pix)[None], reg.w[None], reg.critical, dcfg)
    reasons = []
    if not q_align:
        reasons.append("Q_ALIGN_0")
    if not r["any_val"][0]:
        reasons.append("NO_VALID_REGIONS")
    if not r["req_ok"][0]:
        reasons.append("REQUIRED_REGION_INVISIBLE")
    dec = Decision(int(r["q_cmp"][0]), None, reasons=reasons)
    if r["any_val"][0]:
        dec.D, dec.B, dec.b = float(r["D"][0]), float(r["B"][0]), r["b"][0]
        dec.crit_problem = bool(r["crit_bad"][0])
    if dec.q_cmp:
        dec.violations = [n for n, k in (("D_GLOB", "v_D"), ("B_LOC", "v_B"), ("CRITICAL", "v_C")) if r[k][0]]
        dec.y_pred = int(r["y_pred"][0])
    return dec
