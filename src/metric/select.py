"""Best-configuration selection over tuning runs (spec M-D26). No TensorFlow."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunResult:
    run_id: str
    seed: int
    status: str
    correct_at_1: int | None
    n_valid: int | None
    best_epoch: int | None


@dataclass
class Candidate:
    params: dict
    order: int  # tie-break preference: smaller wins
    runs: list[RunResult]


class SelectError(Exception):
    pass


def select_best(cands: list[Candidate], seeds: list[int]) -> tuple[Candidate, list[dict]]:
    if not cands:
        raise SelectError("no candidates")
    table, n_all = [], set()
    for c in cands:
        ids = ", ".join(r.run_id for r in c.runs)
        bad = [r.run_id for r in c.runs if r.status != "completed" or r.correct_at_1 is None or r.n_valid is None]
        if bad:
            raise SelectError(f"incomplete or failed runs: {', '.join(bad)}")
        if sorted(r.seed for r in c.runs) != sorted(seeds):
            raise SelectError(f"runs [{ids}] do not match seeds {seeds} exactly once each")
        s, n = sum(r.correct_at_1 for r in c.runs), sum(r.n_valid for r in c.runs)
        n_all.update(r.n_valid for r in c.runs)
        table.append({
            "params": c.params, "order": c.order, "sum_correct_at_1": s, "n_valid": n,
            "mean_val_recall_at_1": s / n,
            "runs": [{"run_id": r.run_id, "seed": r.seed, "correct_at_1": r.correct_at_1,
                      "n_valid": r.n_valid, "best_epoch": r.best_epoch} for r in c.runs],
        })
    if len(n_all) != 1:
        raise SelectError(f"n_valid differs between runs: {sorted(n_all)}")
    i = min(range(len(cands)), key=lambda i: (-table[i]["sum_correct_at_1"], cands[i].order))
    return cands[i], table
