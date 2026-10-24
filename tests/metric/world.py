"""A make_world variant for `python -m metric`: real MetricParams, optional bigger SOP and the real p3 builder."""
import copy
import shutil
from pathlib import Path

import yaml

from expcore import registry
from expcore.config import load_config
from expcore.manifests import build_manifests
from expcore.runlog import RunRecord
from tests.cli_helpers import make_world
from tests.fixtures_builder import make_sop
from tests.conftest import write_cfg

REPO = Path(__file__).resolve().parents[2]
SMOKE_PARAMS = yaml.safe_load((REPO / "configs/smoke/metric.yaml").read_text(encoding="utf-8"))["params"]


def metric_world(root: Path, monkeypatch, *, profile="smoke", params=None, method="proxy_cal", sop_classes=None,
                 real_pairs=False, build=True):
    import metric.cli  # noqa: F401  (registers the p3 builder and the params schema)
    from metric.pairs import p3_pair_builder
    algo = "proxy_cos_softmax_v1" if method == "proxy_cal" else f"{method}_v1"
    world = make_world(root, monkeypatch, profile=profile, algorithms={method: algo}, build=False,
                       runs=[{"module": "metric", "command": "train", "config": "cfg.yaml"}])
    if sop_classes:
        sop = Path(world.cfg_dict["paths"]["data_root"]) / "sop"
        shutil.rmtree(sop)
        world.cfg_dict["datasets"]["sop"]["expected"] = make_sop(sop, n_train_classes=sop_classes)
    world.cfg_dict["params"] = copy.deepcopy(params if params is not None else SMOKE_PARAMS)
    world.write()
    if real_pairs:
        monkeypatch.setitem(registry._PAIR_BUILDERS, "p3", p3_pair_builder)
    if build:
        build_manifests(load_config(world.cfg_path, module="metric"), Path(world.cfg_dict["paths"]["data_root"]),
                        Path(world.cfg_dict["paths"]["manifest_dir"]))
    return world


def write_experiment(world, cells: list[dict], name="tune") -> Path:
    """An experiments config listing one train cell per params override (e.g. {"calibration": {"lambda": 0.1}})."""
    runs = []
    for i, over in enumerate(cells):
        d = copy.deepcopy(world.cfg_dict)
        for section, values in over.items():
            d["params"][section].update(values)
        write_cfg(d, world.root / f"{name}-cell{i}.yaml")
        runs.append({"module": "metric", "command": "train", "config": f"{name}-cell{i}.yaml"})
    return write_cfg({**copy.deepcopy(world.cfg_dict), "algorithms": {}, "params": {"runs": runs}},
                     world.root / f"{name}.yaml")


def fake_tuning_runs(exp_path: Path, correct: list[int], *, n_valid=10, status="completed") -> list:
    """Completed (real RunRecord) train run dirs with best_checkpoint.json, one per planned run, in plan order."""
    import experiments.cli  # noqa: F401  (registers the experiments params schema)
    from experiments.plan import make_plan
    from expcore.resume import SUCCESS
    from expcore.io import dump_json
    cfg = load_config(exp_path, module="experiments")
    planned = [r for r, _ in make_plan(cfg, exp_path.parent)]
    for r, c in zip(planned, correct):
        run_dir = cfg.paths.out_root / cfg.profile / "metric" / r.run_id
        rec = RunRecord(run_dir, run_id=r.run_id, config_hash=r.config_hash, dataset_manifest_hash="0" * 64,
                        seed=r.seed, profile=cfg.profile, module="metric", command="train",
                        protocol_version=cfg.protocol_version, oracle_mode=False, post_test_study=False,
                        git_commit=None, dirty_diff_hash=None, environment={}, library_primitives=[])
        rec.start()
        if status == "completed":
            rec.finish("completed")
            (run_dir / SUCCESS).write_bytes(b"")
        else:
            rec.finish("failed", {"type": "OOM", "message": "{}", "unit_id": None})
        dump_json({"epoch": 1, "global_step": 3, "val_recall_at_1": c / n_valid, "val_correct_at_1": c,
                   "n_valid_queries": n_valid, "disclaimers": ["run-level note"]}, run_dir / "best_checkpoint.json")
    return planned
