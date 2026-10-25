"""Training step log (spec 7.2) and per-epoch queue snapshot; numpy/pandas only."""
from pathlib import Path

import numpy as np
import pandas as pd

from expcore.io import write_parquet

STEP_SCHEMA = {
    "run_id": "str", "global_step": "int64", "epoch": "int64", "step_in_epoch": "int64",
    "lambda": "float64", "L_base": "float64", "L_dev": "float64", "L_total": "float64",
    "n_cval": "int64", "n_c_min": "int64", "n_c_mean": "float64", "n_c_max": "int64",
    "n_classes_empty": "int64", "n_classes_full": "int64",
    "p_norm_min": "float64", "p_norm_mean": "float64", "p_norm_max": "float64",
    "mu_norm_min": "float64", "mu_norm_mean": "float64", "mu_norm_max": "float64",
    "mean_age_steps": "float64", "step_time_ms": "float64", "data_time_ms": "float64",
}
STEP_COLUMNS = list(STEP_SCHEMA)
_NULLABLE = ("n_c_mean", "p_norm_min", "p_norm_mean", "p_norm_max", "mu_norm_min", "mu_norm_mean",
             "mu_norm_max", "mean_age_steps")
QUEUE_SCHEMA = {"proxy_idx": "int64", "n_c": "int64", "mean_age": "float64",
                "mu_norm": "float64", "p_norm": "float64"}


def _mmm(prefix, a):
    if a is None or len(a) == 0:
        return {f"{prefix}_min": None, f"{prefix}_mean": None, f"{prefix}_max": None}
    return {f"{prefix}_min": float(a.min()), f"{prefix}_mean": float(a.mean()),
            f"{prefix}_max": float(a.max())}


def step_stats(p, n=None, mu=None, K=None, mean_age=None) -> dict:
    """Queue/proxy statistics of one log row; queue (proxy) fields are null when n (p) is None."""
    row = _mmm("p_norm", None if p is None else np.linalg.norm(np.asarray(p, np.float64), axis=1))
    if n is None:
        row.update(n_c_min=None, n_c_mean=None, n_c_max=None, n_classes_empty=None,
                   n_classes_full=None, mean_age_steps=None, **_mmm("mu_norm", None))
        return row
    n = np.asarray(n)
    mu_n = np.linalg.norm(np.asarray(mu, np.float64)[n > 0], axis=1)
    row.update(n_c_min=int(n.min()), n_c_mean=float(n.mean()), n_c_max=int(n.max()),
               n_classes_empty=int((n == 0).sum()), n_classes_full=int((n >= K).sum()),
               mean_age_steps=mean_age, **_mmm("mu_norm", mu_n))
    return row


def write_steps(rows, path: Path) -> None:
    """rows: list of dicts or a DataFrame with STEP_COLUMNS."""
    write_parquet(pd.DataFrame(rows, columns=STEP_COLUMNS), path, nullable=_NULLABLE,
                  schema=STEP_SCHEMA)


def read_steps(path: Path) -> list[dict]:
    """Rows as written by write_steps; nulls come back as None (not NaN)."""
    df = pd.read_parquet(path)
    return df.astype(object).where(df.notna(), None).to_dict("records")


class StepLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.parts = self.out_dir / "training_steps"
        self.parts.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)

    def end_epoch(self, epoch: int) -> None:
        write_steps(self.rows, self.parts / f"part-{epoch:03d}.parquet")
        self.rows = []

    def finalize(self) -> None:
        df = pd.concat([pd.read_parquet(p) for p in sorted(self.parts.glob("part-*.parquet"))],
                       ignore_index=True)
        write_steps(df, self.out_dir / "training_steps.parquet")


def queue_snapshot(queues, P, step) -> pd.DataFrame:
    """Per-class n_c, mean entry age, centre norm and proxy norm; queue fields null without queues."""
    p_norm = np.linalg.norm(P.numpy().astype(np.float64), axis=1)
    n_cls = len(p_norm)
    if queues is None:
        n = age = mu_norm = [None] * n_cls
    else:
        mu, n = (t.numpy() for t in queues.centers())
        age = queues.age_stats(step)["per_class_age"]
        mu_norm = np.where(n > 0, np.linalg.norm(mu.astype(np.float64), axis=1), np.nan)
    return pd.DataFrame({"proxy_idx": np.arange(n_cls), "n_c": n, "mean_age": age,
                         "mu_norm": mu_norm, "p_norm": p_norm})


def write_queue_snapshot(df: pd.DataFrame, path: Path) -> None:
    write_parquet(df, path, nullable=("mean_age", "mu_norm"), schema=QUEUE_SCHEMA)
