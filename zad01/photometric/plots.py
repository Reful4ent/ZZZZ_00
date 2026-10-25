"""Диагностические кривые (matplotlib, без интерактивного backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FACTOR_LABELS = {
    "brightness": "сдвиг яркости b",
    "contrast": "коэффициент контраста a",
    "gamma": "γ",
    "noise": "σ гауссова шума",
    "jpeg": "JPEG quality",
}


def e2_curves(curves: pd.DataFrame, out_dir: Path, split: str = "test") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    c = curves[curves["split"] == split]
    methods = sorted(c["method_id"].unique())
    for factor, g in c.groupby("factor"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
        for mid in methods:
            m = g[g["method_id"] == mid].sort_values("param")
            for ax, col in zip(axes, ["recall_all", "fpr_all", "coverage"]):
                ax.errorbar(m["param"], m[f"{col}_mean"], yerr=m[f"{col}_std"].fillna(0), marker="o", ms=3, capsize=2, label=mid)
        for ax, title in zip(axes, ["Recall_all", "FPR_all", "Coverage"]):
            ax.set_title(title)
            ax.set_xlabel(FACTOR_LABELS.get(factor, factor))
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
        axes[2].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"P1-E2: {FACTOR_LABELS.get(factor, factor)} (split={split}, фиксированный порог из validation P1-E1)")
        fig.tight_layout()
        p = out_dir / f"e2_{factor}_decisions.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p)

        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
        for ax, mid in zip(axes.ravel(), methods):
            m = g[g["method_id"] == mid].sort_values("param")
            ax.errorbar(m["param"], m["score_pos_mean_mean"], yerr=m["score_pos_mean_std"].fillna(0), marker="o", ms=3, label="положительные")
            ax.errorbar(m["param"], m["score_neg_mean_mean"], yerr=m["score_neg_mean_std"].fillna(0), marker="s", ms=3, label="отрицательные")
            ax.set_title(mid, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(methods):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle(f"P1-E2: средний score (Ф-5) — {FACTOR_LABELS.get(factor, factor)}, split={split}")
        fig.tight_layout()
        p = out_dir / f"e2_{factor}_scores.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p)
    return files


def e1_by_level(by_type_level: pd.DataFrame, out_dir: Path, split: str = "test") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = by_type_level[by_type_level["split"] == split]
    types = sorted(d["type_id"].unique())
    ncol = 4
    nrow = -(-len(types) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), sharey=True)
    for ax, t in zip(axes.ravel(), types):
        g = d[d["type_id"] == t]
        for mid, m in g.groupby("method_id"):
            ax.plot(m["level"], m["recall_all"], marker="o", ms=3, label=mid)
        ax.set_title(f"{t} {g['type_name'].iloc[0]}", fontsize=9)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(types):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle(f"P1-E1: Recall_all по уровню KADID (split={split})")
    fig.tight_layout()
    p = out_dir / "e1_recall_by_level.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
