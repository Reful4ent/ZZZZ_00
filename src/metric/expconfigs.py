"""P3-E1..E7 experiment configs (spec 10, 8.4), generated from configs/templates/metric.yaml. TF-free.

An experiment = one `experiments` config (params.runs) listing cell configs (one metric config per grid cell);
`experiments plan` expands the cells over the profile's seeds. Selection-dependent experiments take `selected`:
{experiment_id: parsed selected_config.json}, e.g. selected["e1_tune"]["selected_params"] == {"lambda": 0.1}
(flat varied keys: lambda, K, n_warm, lr, margin). e5 also needs N_cohort_all (`n_cohort_images`, or
selected["n_cohort_images"]); e7 uses selected["e1_tune"] and any of selected["e4_contrastive|e4_triplet|e4_classifier"].

Confirmatory: plan only "confirmatory" (all of CONFIRMATORY in one experiments config). expcore keeps one
plan.lock.json per (profile, protocol_version) and freezes it when test opens, so planning e1_confirm, e4_* or
e5_limited separately would replace each other's lock. e7_degrade cells are `search` stages that need
--gallery/--queries: they are not plan entries and run manually after all confirmatory training. In the combined
plan e4_fixed's deterministic cells are expanded over all 5 confirmatory seeds (make_plan takes the plan's seeds).
"""
import copy
import itertools
import math
from pathlib import Path

import yaml

from expcore.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
SEEDS = {"pilot": [11, 23, 37], "confirmatory": [11, 23, 37, 51, 71]}
SECTION = {"lambda": "calibration", "K": "calibration", "n_warm": "calibration", "queues": "calibration",
           "lr": "optimizer", "margin": "loss", "train_augment": "input", "per_class_limit": "data",
           "cohort_min_images": "data", "steps_per_epoch": "train", "degradation": None}
UNTRAINED = ("imagenet_fixed", "rgb_hist")
DEGRADATIONS = ("bright_+0.1", "bright_-0.1", "noise_0.03")
SELECTION_FREE = ("smoke", "e1_tune", "e4_contrastive", "e4_triplet", "e4_classifier", "e4_fixed", "e6_nol2",
                  "base_minimal")
CONFIRMATORY = ("e1_confirm", "e4_confirm", "e4_fixed", "e5_limited")


def _grid(method, command="train", fixed=None, **axes):
    return [(method, command, {**(fixed or {}), **dict(zip(axes, vals))}) for vals in itertools.product(*axes.values())]


def _selected(selected, exp):
    if not selected or exp not in selected:
        raise ConfigError(f"this experiment needs selected[{exp!r}] (selected_config.json of that tuning experiment)")
    return selected[exp]


def _lambda_star(selected):
    return _selected(selected, "e1_tune")["selected_params"]["lambda"]


def _lambda_sensitivity(selected):
    """E2/E3 lambda: lambda*, or (M-D20, lambda*=0) the best lambda>0 among E1's candidates, tie -> smaller."""
    lam = _lambda_star(selected)
    if lam > 0:
        return lam
    pos = [c for c in selected["e1_tune"]["candidates"] if c["params"]["lambda"] > 0]
    if not pos:
        raise ConfigError("e1_tune candidates contain no lambda>0")
    return max(pos, key=lambda c: (c["mean_val_recall_at_1"], -c["params"]["lambda"]))["params"]["lambda"]


def _e4_models(selected):
    return [(e4[3:], "train", _selected(selected, e4)["selected_params"])
            for e4 in ("e4_contrastive", "e4_triplet", "e4_classifier")]


def _cells(exp, selected, n_cohort):
    """(profile, [(method_id, command, flat overrides)])."""
    if exp == "e1_tune":
        return "pilot", _grid("proxy_cal", **{"lambda": (0, 0.01, 0.1, 1)})
    if exp == "e1_confirm":
        return "confirmatory", _grid("proxy_cal", **{"lambda": sorted({_lambda_star(selected), 0}, reverse=True)})
    if exp == "e2_k":
        return "pilot", _grid("proxy_cal", fixed={"lambda": _lambda_sensitivity(selected)}, K=(8, 32, 128))
    if exp == "e3_nwarm":
        return "pilot", _grid("proxy_cal", fixed={"lambda": _lambda_sensitivity(selected)}, n_warm=(1, 4, 8))
    if exp == "e4_contrastive":
        return "pilot", _grid("contrastive", lr=(1e-5, 1e-4), margin=(0.5, 1))
    if exp == "e4_triplet":
        return "pilot", _grid("triplet", lr=(1e-5, 1e-4), margin=(0.1, 0.2))
    if exp == "e4_classifier":
        return "pilot", _grid("classifier", lr=(1e-5, 1e-4, 3e-4, 1e-3))
    if exp == "e4_fixed":   # untrained: one deterministic run on the split (seed 11 recorded for the contract)
        return "confirmatory", [(m, "embed", {}) for m in UNTRAINED]
    if exp == "e4_confirm":
        return "confirmatory", _e4_models(selected)
    if exp == "e5_limited":
        if not n_cohort:
            n_cohort = (selected or {}).get("n_cohort_images")
        if not n_cohort:
            raise ConfigError("e5_limited needs n_cohort_images (N_cohort_all, classes with >=4 images; M-D21)")
        return "confirmatory", _grid("proxy_cal", fixed={"cohort_min_images": 4, "steps_per_epoch": math.ceil(n_cohort / 32)},
                                     **{"lambda": sorted({_lambda_star(selected), 0}, reverse=True),
                                        "per_class_limit": (2, 3, 4, None)})
    if exp == "e6_noaug":
        return "pilot", _grid("proxy_cal", fixed={"lambda": _lambda_star(selected), "train_augment": False})
    if exp == "e6_nol2":   # M-D22: lambda=0, no queues
        return "pilot", _grid("proxy_softmax_noL2", fixed={"lambda": 0, "queues": False})
    if exp == "base_minimal":
        return "pilot", _grid("base_minimal", fixed={"lambda": 0, "queues": False})
    if exp == "e7_degrade":
        models = _grid("proxy_cal", **{"lambda": sorted({_lambda_star(selected), 0}, reverse=True)})
        models += _e4_models(selected)
        models += [(m, "train", {}) for m in UNTRAINED]
        return "confirmatory", [(m, "search", {**o, "degradation": d}) for m, _, o in models for d in DEGRADATIONS]
    raise ConfigError(f"unknown experiment {exp!r}")


def _name(method, over):
    return "-".join([method] + [f"{k}{'all' if v is None else v}" for k, v in over.items()])


def _dump(d):
    return yaml.safe_dump(d, allow_unicode=True, sort_keys=True)


def _write_cells(exp_id, out_dir, selected, n_cohort_images):
    """(base config, experiments runs, cell paths) of one experiment; writes cells/<exp_id>/*.yaml."""
    if exp_id == "smoke":   # one train run of the smoke config
        profile, cells = "smoke", [("proxy_cal", "train", {})]
        base = yaml.safe_load((REPO / "configs/smoke/metric.yaml").read_text(encoding="utf-8"))
    else:
        profile, cells = _cells(exp_id, selected, n_cohort_images)
        base = yaml.safe_load((REPO / "configs/templates/metric.yaml").read_text(encoding="utf-8"))
    seeds = [11] if profile == "smoke" or exp_id == "e4_fixed" or exp_id == "base_minimal" else SEEDS[profile]
    base |= {"profile": profile, "seeds": {"seed": seeds[0], "seeds": seeds}}
    paths, runs = [], []
    for method, command, over in cells:
        cell = copy.deepcopy(base)
        cell["algorithms"] = {method: "proxy_cos_softmax_v1" if method == "proxy_cal" else f"{method}_v1"}
        for k, v in over.items():
            if SECTION[k]:
                cell["params"][SECTION[k]][k] = v
            else:
                cell["params"][k] = v
        rel = f"cells/{exp_id}/{command}-{_name(method, over)}.yaml"
        paths.append(out_dir / rel)
        paths[-1].parent.mkdir(parents=True, exist_ok=True)
        paths[-1].write_bytes(_dump(cell).encode())
        runs.append({"module": "metric", "command": command, "config": rel})
    return base, runs, paths


def materialize(exp_id: str, out_dir: Path, selected: dict | None = None, n_cohort_images: int | None = None) -> list[Path]:
    """Write <out_dir>/<exp_id>.yaml (experiments config) and <out_dir>/cells/<exp>/*.yaml; experiments config first.

    exp_id "confirmatory": one experiments config over the cells of every CONFIRMATORY experiment.
    """
    out_dir = Path(out_dir)
    paths, runs = [out_dir / f"{exp_id}.yaml"], []
    for sub in CONFIRMATORY if exp_id == "confirmatory" else (exp_id,):
        base, sub_runs, sub_paths = _write_cells(sub, out_dir, selected, n_cohort_images)
        runs += sub_runs
        paths += sub_paths
    if exp_id == "confirmatory":
        base |= {"seeds": {"seed": 11, "seeds": SEEDS["confirmatory"]}}
    paths[0].write_bytes(_dump({**base, "algorithms": {}, "params": {"runs": runs}}).encode())
    return paths
