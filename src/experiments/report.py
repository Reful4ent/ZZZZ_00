"""experiments report (spec §1.6, §6.2, §7.7, ПОСТ-02): every completed planned run, whatever its result."""

import json
import math
from typing import Sequence

import pandas as pd

from expcore.errors import ProtocolViolationError
from expcore.reporting import DISCLAIMER, dataset_display_name
from experiments.plan import PlannedRun

METRIC_LABELS = [
    ("recall_all", "Recall_all"), ("fpr_all", "FPR_all"), ("correct_all", "Correct_all"),
    ("coverage", "Coverage"), ("coverage_pos", "Coverage+"), ("coverage_neg", "Coverage−"), ("rho_rej", "rho_rej"),
    ("recall_cond", "Recall_cond (conditional)"), ("fpr_cond", "FPR_cond (conditional)"),
    ("correct_cond", "Correct_cond (conditional)"),
]
SEED_SPREAD = ["recall_all", "fpr_all", "correct_all", "coverage"]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt(v, reason: str | None = None) -> str:
    if _missing(v):
        return f"null ({reason})" if reason else "null"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _matrix(row: pd.Series) -> list[str]:
    lines = ["| Истинный класс | y_pred=1 | y_pred=0 | reject | Всего |", "|---|---|---|---|---|"]
    for label, cells in ((f"{row.positive_class} (positive)", (row.tp, row.fn, row.rej_pos)),
                         ("negative", (row.fp, row.tn, row.rej_neg))):
        if any(_missing(c) for c in cells):
            lines.append(f"| {label} | — | — | — | — |")
            continue
        total = int(sum(cells))
        shares = [f"{int(c)} ({c / total:.2f})" if total else f"{int(c)}" for c in cells]
        lines.append(f"| {label} | " + " | ".join(shares) + f" | {total} |")
    return lines


def _run_block(row: pd.Series) -> list[str]:
    reasons = json.loads(row.null_reasons) if row.null_reasons else {}
    lines = [f"### {row.method_id} · {row.relation_id} · {row.split} · seed {row.seed}", "",
             f"Запуск `{row.run_id}`; n_pairs = {row.n_pairs}, n_source_groups = {row.n_source_groups}, "
             f"n_seeds = {row.n_seeds}, n_not_applicable = {_fmt(row.n_not_applicable)}.", "",
             f"Положительный класс: {row.positive_class} (y_match = 1); преобразование оценки: "
             f"{row.score_transform}. При смене класса (y_defect = 1 − y_match) score_defect = −score_match, "
             "отказы остаются отказами.", ""]
    if all(_missing(row[k]) for k, _ in METRIC_LABELS):
        reason = next(iter(reasons.values()), "undefined")
        return lines + [f"Метрики этого запуска не вычисляются ({reason}).", ""]
    lines += ["| Метрика | Значение |", "|---|---|"]
    lines += [f"| {label} | {_fmt(row[k], reasons.get(k))} |" for k, label in METRIC_LABELS]
    return lines + [""] + _matrix(row) + [""]


def _section(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["Нет запусков.", ""]
    out = []
    for _, row in rows.iterrows():
        out += _run_block(row)
    return out


def _seed_spread(rows: pd.DataFrame) -> list[str]:
    lines = ["| Метод | Отношение | Split | n_seeds | " + " | ".join(f"{m} (mean / min / max)" for m in SEED_SPREAD)
             + " |", "|---|---|---|---|" + "---|" * len(SEED_SPREAD)]
    for (method, relation, split), g in rows.groupby(["method_id", "relation_id", "split"], sort=True):
        cells = []
        for m in SEED_SPREAD:
            v = g[m].dropna()
            cells.append("null" if v.empty else f"{v.mean():.4f} / {v.min():.4f} / {v.max():.4f}")
        lines.append(f"| {method} | {relation} | {split} | {g['seed'].nunique()} | " + " | ".join(cells) + " |")
    return lines + ["", "Повторы по seeds оценивают изменчивость алгоритма на тех же изображениях и не увеличивают "
                        "число независимых наблюдений (§7.1).", ""]


def render_report(metrics: pd.DataFrame, plan: Sequence[PlannedRun], completeness: dict) -> str:
    """Markdown report; a completed planned run missing from `metrics` → ProtocolViolationError (ПОСТ-02)."""
    statuses = completeness.get("runs", {})
    completed = [r.run_id for r in plan if statuses.get(r.run_id) == "completed"]
    missing = sorted(set(completed) - set(metrics["run_id"]))
    if missing:
        raise ProtocolViolationError(f"completed planned runs missing from the report: {missing}")
    rows = metrics.sort_values(["module", "method_id", "relation_id", "split", "seed"], kind="mergesort")
    lines = ["# Отчёт эксперимента", "", f"> {DISCLAIMER}", "", "## Наборы данных", ""]
    lines += [f"- {dataset_display_name(d, c)}" for d, c in sorted(completeness.get("datasets", {}).items())]
    lines += ["", "## Запуски плана", "", "| run_id | Модуль | Команда | seed | Статус |", "|---|---|---|---|---|"]
    lines += [f"| `{r.run_id}` | {r.module} | {r.command} | {r.seed} | {statuses.get(r.run_id, 'not started')} |"
              for r in plan]
    main, oracle = rows[rows["section"] == "main"], rows[rows["section"] == "oracle"]
    lines += ["", "## Результаты", "", "Показатели `*_all` — основные; `*_cond` — условные (только по принятым "
              "решениям) и не заменяют `*_all`.", ""]
    lines += _section(main)
    lines += ["## Изменчивость по seeds", ""] + _seed_spread(main)
    lines += ["## Oracle-запуски", "", "Запуски в режиме oracle (доступ к оценочным данным) — отдельная секция, "
              "в подтверждающие таблицы не входят.", ""] + _section(oracle)
    return "\n".join(lines) + "\n"
