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
        odd = [r.run_id for r in c.runs if r.n_valid < 1 or not 0 <= r.correct_at_1 <= r.n_valid]
        if odd:
            raise SelectError(f"runs with n_valid < 1 or correct_at_1 outside [0, n_valid]: {', '.join(odd)}")
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


VARIED = (("calibration", "lambda"), ("calibration", "K"), ("calibration", "n_warm"), ("optimizer", "lr"),
          ("loss", "margin"))   # flat keys of selected_params, as metric.expconfigs consumes them
TIE_RULE = ("max mean validation Recall@1 over seeds = sum(val_correct_at_1) / sum(n_valid_queries); "
            "tie -> earlier grid cell (grid order, smaller lambda first for P3-E1)")


def _status(run_dir) -> str:
    from expcore.io import load_json
    from expcore.resume import SUCCESS
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return "not started"
    status = load_json(run_json)["status"]
    return status if status != "completed" or (run_dir / SUCCESS).exists() else "incomplete"


def select(exp_config, path_overrides=None):
    """Spec 10.1: selected_config.json of the tuning grid listed by an experiments config; returns its path.

    Written to out_root/<profile>/metric/select/<experiments config stem>/. Any failed or missing run aborts
    (failed_runs_policy: abort) with the list of run_ids.
    """
    from pathlib import Path

    import yaml

    import experiments.cli  # noqa: F401  (registers the experiments params schema)
    from experiments.plan import make_plan
    from expcore.config import config_hash, load_config
    import json

    from expcore.errors import DataValidationError, ProtocolViolationError
    from expcore.io import dump_json, load_json
    from expcore.runlog import now_utc
    from metric.disclaimers import ALWAYS

    exp_config = Path(exp_config)
    cfg = load_config(exp_config, module="experiments", path_overrides=path_overrides)
    planned = [r for r, _ in make_plan(cfg, exp_config.parent)]
    seeds, n = list(cfg.seeds.seeds), len(cfg.seeds.seeds)
    flat = []
    for entry in cfg.params.runs:
        p = yaml.safe_load((exp_config.parent / entry.config).read_text(encoding="utf-8"))["params"]
        flat.append({k: p[s][k] for s, k in VARIED})
    keys = [k for _, k in VARIED if len({f[k] for f in flat}) > 1] or [k for _, k in VARIED]
    cands, notes, bad = [], list(ALWAYS), []
    for i, f in enumerate(flat):
        runs = []
        for r in planned[i * n:(i + 1) * n]:
            run_dir = cfg.paths.out_root / cfg.profile / "metric" / r.run_id
            status = _status(run_dir)
            best = load_json(run_dir / "best_checkpoint.json") if (run_dir / "best_checkpoint.json").exists() else {}
            if status != "completed" or not best:
                bad.append(f"{r.run_id} ({status})")
            notes += [t for t in best.get("disclaimers", []) if t not in notes]
            runs.append(RunResult(r.run_id, r.seed, status, best.get("val_correct_at_1"),
                                  best.get("n_valid_queries"), best.get("epoch")))
        cands.append(Candidate({k: f[k] for k in keys}, i, runs))
    if bad:
        raise DataValidationError(f"select: incomplete or failed tuning runs (failed_runs_policy: abort): "
                                  f"{', '.join(bad)}")
    try:
        best, table = select_best(cands, seeds)
    except SelectError as e:
        raise DataValidationError(f"select: {e}") from e
    entry = cfg.params.runs[best.order]
    out = cfg.paths.out_root / cfg.profile / "metric" / "select" / exp_config.stem / "selected_config.json"
    doc = {
        "experiment_id": exp_config.stem, "protocol_version": cfg.protocol_version, "profile": cfg.profile,
        "grid": [c.params for c in cands],
        "candidates": [{"params": t["params"], "mean_val_recall_at_1": t["mean_val_recall_at_1"],
                        "runs": [{"run_id": r["run_id"], "seed": r["seed"], "val_correct_at_1": r["correct_at_1"],
                                  "n_valid_queries": r["n_valid"], "best_epoch": r["best_epoch"]}
                                 for r in t["runs"]]} for t in table],
        "tie_rule": TIE_RULE, "selected_params": best.params, "selected_config": entry.config,
        "selected_config_hash": config_hash(load_config(exp_config.parent / entry.config, module="metric")),
        "created_at": now_utc(), "disclaimers": notes,
    }
    if out.exists():   # write-once: a selection is never silently replaced
        old = load_json(out)
        if {k: v for k, v in old.items() if k != "created_at"} !=                 json.loads(json.dumps({k: v for k, v in doc.items() if k != "created_at"})):
            raise ProtocolViolationError(f"{out}: a different selection is already recorded")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(doc, out)
    return out
