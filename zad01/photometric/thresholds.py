"""Выбор порога (Г2-167): t* ∈ argmax Recall_val(t) при FPR_val(t) ≤ α.

Кандидаты — уникальные validation-score и крайние пороги (ниже минимума и выше
максимума); правило решения y = 1[s ≥ t]. Recall_all и FPR_all считаются по всем
парам, undefined-оценки — отказы (решение 0), но остаются в знаменателях.
Тай-брейки: больший Recall → меньший FPR → более строгий (больший) порог.
Если ни один порог с непустым набором положительных решений не допустим,
выбирается порог выше максимального validation-score (fallback, Recall = 0).
"""

from __future__ import annotations

import math

import numpy as np


def select_threshold(scores, labels, alpha: float) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("threshold selection needs both positive and negative validation pairs")
    defined = np.isfinite(s)
    sd, yd = s[defined], y[defined]

    if sd.size == 0:
        return _result(math.inf, 0, 0, n_pos, n_neg, 0, True, s.size, "no defined validation scores")

    uniq = np.unique(sd)  # по возрастанию
    above_max = float(np.nextafter(uniq[-1], math.inf))
    # Число пар с s ≥ t для каждого t из uniq.
    pos_sorted = np.sort(sd[yd])
    neg_sorted = np.sort(sd[~yd])
    tp = pos_sorted.size - np.searchsorted(pos_sorted, uniq, side="left")
    fp = neg_sorted.size - np.searchsorted(neg_sorted, uniq, side="left")
    # Нижний крайний порог (−inf) даёт те же решения, что uniq[0], и проигрывает ему
    # по тай-брейку «более строгий»; верхний — пустой набор положительных решений.
    cand_t = np.append(uniq, above_max)
    cand_tp = np.append(tp, 0)
    cand_fp = np.append(fp, 0)
    recall = cand_tp / n_pos
    fpr = cand_fp / n_neg

    feasible = fpr <= alpha
    nonempty = (cand_tp + cand_fp) > 0
    if not (feasible & nonempty).any():
        return _result(above_max, 0, 0, n_pos, n_neg, sd.size, True, cand_t.size,
                       "no non-empty decision set satisfies FPR <= alpha; Recall = 0")

    # Лексикографический выбор: max recall, min fpr, max t.
    idx = np.flatnonzero(feasible)
    order = np.lexsort((-cand_t[idx], fpr[idx], -recall[idx]))
    best = idx[order[0]]
    fallback = bool(best == cand_t.size - 1)
    note = "selected threshold is above max validation score; Recall = 0" if fallback else ""
    return _result(float(cand_t[best]), int(cand_tp[best]), int(cand_fp[best]), n_pos, n_neg, sd.size, fallback, cand_t.size, note)


def _result(t, tp, fp, n_pos, n_neg, n_defined, fallback, n_candidates, note) -> dict:
    return {
        "threshold": t,
        "rule": "decision = 1[score >= threshold]",
        "recall_all": tp / n_pos,
        "fpr_all": fp / n_neg,
        "coverage": n_defined / (n_pos + n_neg),
        "tp": tp,
        "fp": fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_defined": n_defined,
        "n_candidates": n_candidates,
        "fallback_above_max": bool(fallback),
        "note": note,
    }
