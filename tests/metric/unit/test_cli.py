"""metric CLI (Task 7.1): commands, exit codes, prepare_run vs expcore.execute_run, select, disclaimers."""
import dataclasses
import subprocess
import sys

import pytest

from expcore.cli_common import command_parser, execute_run
from expcore.errors import NotImplementedAlgorithm
from expcore.io import load_json
from expcore.manifests import load_manifest, manifest_hash
from tests.metric.world import REPO, SMOKE_PARAMS, fake_tuning_runs, metric_world, write_experiment

pytestmark = pytest.mark.unit

COMMANDS = ("train", "resume", "embed", "search", "select")
SPEC_DISCLAIMERS = (   # spec 1.6 items 2-6, verbatim
    "Базовый критерий — косинусная softmax cross-entropy (Г2-150), `loss_id=proxy_cos_softmax_v1`; это "
    "конкретизация общей модели главы 2, а не подтверждённая копия функции авторской рукописи.",
    "Численные настройки P3 — начальный проектный профиль, а не настройки, восстановленные из статьи.",
    "Эпоха — назначенное число равномерно семплированных пакетов, а не гарантированный однократный проход "
    "каждого изображения.",
    "Снижение Ldev само по себе не является успешным результатом.",
    "Выбор λ по Recall@1 не означает оптимальности бинарного порога.",
)


def _main(argv):
    from metric.cli import main
    return main(argv)


def _train_run_dir(world):
    from expcore.config import config_hash, load_config
    from expcore.runlog import make_run_id
    cfg = load_config(world.cfg_path, module="metric")
    return world.out_root / cfg.profile / "metric" / make_run_id("metric", cfg.profile, config_hash(cfg), 11)


@pytest.mark.parametrize("cmd", COMMANDS)
def test_help_all_commands(cmd):   # T-M-CLI-01
    assert _main([cmd, "--help"]) == 0


def test_tf_still_lazy():
    code = "import sys, metric, metric.cli; assert 'tensorflow' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=REPO).returncode == 0


def test_pair_builder_registered(tmp_path, monkeypatch):
    code = ("import experiments.cli; from expcore.registry import pair_builders; "   # build-manifests' process
            "from metric.pairs import p3_pair_builder; assert pair_builders()['p3'] is p3_pair_builder")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=REPO).returncode == 0
    world = metric_world(tmp_path, monkeypatch, real_pairs=True)
    m = load_manifest(world.out_root / "manifests", "v1")
    pairs = m.pairs["p3"]
    assert len(pairs) and pairs["pair_id"].str.startswith("p3:").all()
    assert m.manifest_hash == manifest_hash(m.images, m.datasets, m.pairs)
    assert m.manifest_hash != manifest_hash(m.images, m.datasets, {k: v for k, v in m.pairs.items() if k != "p3"})


def test_prepare_run_matches_execute_run(tmp_path, monkeypatch):
    from metric.runctx import prepare_run
    world = metric_world(tmp_path, monkeypatch)
    args = command_parser("metric", ["train"]).parse_args(world.argv("train"))
    with pytest.raises(NotImplementedAlgorithm):   # expcore writes a failed run.json with its own RunRecord
        execute_run("metric", "train", args, query=True)
    run_dir = _train_run_dir(world)
    before = load_json(run_dir / "run.json")
    run = prepare_run("train", args, deterministic_ops=None)
    assert (run.run_id, run.run_dir) == (before["run_id"], run_dir)
    run.record.start()   # identity check against expcore's run.json (ResultTamperedError on any difference)
    after = load_json(run_dir / "run.json")
    same = ("run_id", "config_hash", "dataset_manifest_hash", "seed", "profile", "module", "command",
            "protocol_version", "oracle_mode", "post_test_study", "git_commit", "dirty_diff_hash",
            "library_primitives", "paths_resolved", "schema_version")
    assert {k: after[k] for k in same} == {k: before[k] for k in same}
    assert len(after["attempts"]) == 2


def test_confirmatory_outside_plan_exit_5(tmp_path, monkeypatch):
    from metric.params import BASE_PARAMS
    world = metric_world(tmp_path, monkeypatch, profile="confirmatory", params=BASE_PARAMS)
    assert _main(world.argv("train")) == 5


def test_leakage_exit_code(tmp_path, monkeypatch):
    import metric.runctx as runctx
    world = metric_world(tmp_path, monkeypatch)
    real = runctx.load_manifest

    def leaky(*a, **kw):
        m = real(*a, **kw)
        im = m.images.copy()
        sop = im["dataset_id"] == "sop"
        train_cls = im.loc[sop & (im["split"] == "train"), "class_id"].iloc[0]
        im.loc[im.index[sop & (im["split"] == "validation")][0], "class_id"] = train_cls
        return dataclasses.replace(m, images=im)

    monkeypatch.setattr(runctx, "load_manifest", leaky)
    assert _main(world.argv("train")) == 3


@pytest.mark.parametrize("mode", ["raise", "result"])
def test_runstop_exit_code(tmp_path, monkeypatch, mode):
    import metric.train as train
    from metric.errors import RunStop
    world = metric_world(tmp_path, monkeypatch)

    def fake(ctx):
        if mode == "raise":
            raise RunStop("SAMPLER_ERROR", {"n_eligible": 3})
        return {"status": "failed", "failure": {"type": "QUEUE_SUM_DRIFT", "details": {"global_step": 7}},
                "queue_checks": [{"global_step": 5, "max_abs_diff": 0.0, "sum_tol": 1e-6}]}

    monkeypatch.setattr(train, "run_training", fake)
    assert _main(world.argv("train")) == 4
    run_dir = _train_run_dir(world)
    run = load_json(run_dir / "run.json")
    ftype = "SAMPLER_ERROR" if mode == "raise" else "QUEUE_SUM_DRIFT"
    assert run["status"] == "failed" and run["failure"]["type"] == ftype
    assert ftype in (run_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert not (run_dir / "_SUCCESS").exists()
    if mode == "result":
        assert "queue_check" in (run_dir / "run.log.jsonl").read_text(encoding="utf-8")


def test_untrainable_train_refused(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch, method="rgb_hist")
    assert _main(world.argv("train")) == 2


def test_disclaimers_present():
    from metric.disclaimers import disclaimers
    from expcore.reporting import DISCLAIMER
    p = dict(SMOKE_PARAMS, data=dict(SMOKE_PARAMS["data"]))
    d = disclaimers("proxy_cal", p)
    assert DISCLAIMER in d[0] and all(t in d for t in SPEC_DISCLAIMERS) and len(d) == 6
    assert len(disclaimers("base_minimal", p)) == 7
    p["data"].update(per_class_limit=2, class_subset={"n_classes": 20, "seed": 1})
    d = disclaimers("proxy_cal", p, subset=(20, 60, 8))
    assert len(d) == 8 and "20" in d[-1] and "60" in d[-1] and "8" in d[-1]


def _select(world, correct, **kw):
    exp = write_experiment(world, [{"calibration": {"lambda": 0.0}}, {"calibration": {"lambda": 0.1}}])
    planned = fake_tuning_runs(exp, correct, **kw)
    code = _main(["select", "--config", str(exp)])
    out = world.out_root / "smoke" / "metric" / "select" / "tune" / "selected_config.json"
    return code, out, planned


def test_select_picks_best_and_writes_spec_fields(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch, build=False)
    code, out, planned = _select(world, [4, 6])
    assert code == 0
    sel = load_json(out)
    assert {"experiment_id", "protocol_version", "grid", "candidates", "tie_rule", "selected_params",
            "selected_config_hash", "created_at", "disclaimers"} <= set(sel)
    assert sel["selected_params"] == {"lambda": 0.1} and sel["experiment_id"] == "tune"
    assert [c["params"] for c in sel["candidates"]] == [{"lambda": 0.0}, {"lambda": 0.1}]
    run = sel["candidates"][1]["runs"][0]
    assert run == {"run_id": planned[1].run_id, "seed": 11, "val_correct_at_1": 6, "n_valid_queries": 10,
                   "best_epoch": 1}
    assert sel["candidates"][1]["mean_val_recall_at_1"] == 0.6
    assert all(t in sel["disclaimers"] for t in SPEC_DISCLAIMERS) and "run-level note" in sel["disclaimers"]


def test_select_tie_takes_smaller_lambda(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch, build=False)
    code, out, _ = _select(world, [5, 5])
    assert code == 0 and load_json(out)["selected_params"] == {"lambda": 0.0}


def test_select_refuses_failed_runs(tmp_path, monkeypatch, capsys):
    world = metric_world(tmp_path, monkeypatch, build=False)
    code, out, planned = _select(world, [5, 5], status="failed")
    assert code == 3 and not out.exists()
    assert planned[0].run_id in capsys.readouterr().err


def _npz(world, name, *, method="proxy_cal", wh="a" * 64, split="validation", degradation="none"):
    import numpy as np
    from metric.embed import EmbeddingSet, save_embeddings
    m = load_manifest(world.out_root / "manifests", "v1")
    ids = sorted(m.images.loc[(m.images.dataset_id == "sop") & (m.images.split == split), "image_id"])
    z = np.eye(len(ids), 4, dtype=np.float32) + 0.5
    path = world.root / name
    save_embeddings(path, EmbeddingSet(z / np.linalg.norm(z, axis=1, keepdims=True), np.array(ids), wh, split,
                                       degradation, method, m.manifest_hash))
    return str(path)


def _stage_dirs(world):
    return list((world.out_root / "smoke" / "metric").glob("*/stage.json"))


def test_search_refuses_mixed_encoders(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch)
    g, q = _npz(world, "g.npz"), _npz(world, "q.npz", wh="b" * 64)
    assert _main(world.argv("search", "--gallery", g, "--queries", q)) == 2
    assert not _stage_dirs(world)


def test_search_pairs_refused_for_raw_l2(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch, method="proxy_softmax_noL2")
    g = _npz(world, "g.npz", method="proxy_softmax_noL2")
    assert _main(world.argv("search", "--gallery", g, "--queries", g, "--pairs")) == 2
    assert not _stage_dirs(world)


def test_untrainable_embed_is_the_planned_base_run(tmp_path, monkeypatch):
    import experiments.cli  # noqa: F401
    from expcore.config import load_config
    from experiments.plan import make_plan
    from tests.conftest import write_cfg
    world = metric_world(tmp_path, monkeypatch, method="rgb_hist")
    exp = write_cfg({**world.cfg_dict, "algorithms": {},
                     "params": {"runs": [{"module": "metric", "command": "embed", "config": "cfg.yaml"}]}},
                    tmp_path / "e4_fixed.yaml")
    (planned, text), = make_plan(load_config(exp, module="experiments"), tmp_path)
    cell = tmp_path / "planned.yaml"
    cell.write_text(text, encoding="utf-8")
    assert _main(["embed", "--config", str(cell), "--protocol-registry", str(world.registry_path)]) == 0
    run_dir = world.out_root / "smoke" / "metric" / planned.run_id
    assert (run_dir / "_SUCCESS").exists() and load_json(run_dir / "run.json")["command"] == "embed"
    assert (run_dir / "embeddings-validation.npz").exists() and (run_dir / "embeddings-test.npz").exists()


def test_select_is_write_once(tmp_path, monkeypatch):
    world = metric_world(tmp_path, monkeypatch, build=False)
    code, out, planned = _select(world, [4, 6])
    assert code == 0 and _main(["select", "--config", str(world.root / "tune.yaml")]) == 0   # same: no-op
    best = world.out_root / "smoke" / "metric" / planned[0].run_id / "best_checkpoint.json"
    best.write_text(best.read_text(encoding="utf-8").replace('"val_correct_at_1": 4', '"val_correct_at_1": 9'),
                    encoding="utf-8")
    assert _main(["select", "--config", str(world.root / "tune.yaml")]) == 5
    assert load_json(out)["selected_params"] == {"lambda": 0.1}


def test_search_pairs_refuse_degraded_queries(tmp_path, monkeypatch):   # spec 9.7: degradations are query-only
    world = metric_world(tmp_path, monkeypatch)
    g, q = _npz(world, "g.npz"), _npz(world, "q.npz", degradation="noise_0.03")
    assert _main(world.argv("search", "--gallery", g, "--queries", q, "--pairs")) == 2
    assert not _stage_dirs(world)


def test_weights_checksum_measured(tmp_path, monkeypatch):   # spec 5.1
    import metric.model
    import metric.train as train
    from metric.params import BASE_PARAMS, WEIGHTS_SHA256
    world = metric_world(tmp_path, monkeypatch, profile="pilot", params=BASE_PARAMS)
    seen = []
    monkeypatch.setattr(train, "run_training", lambda ctx: seen.append(ctx.weights_sha256) or {"status": "completed"})
    monkeypatch.setattr(metric.model, "weights_file_sha256", lambda download=False: "0" * 64)
    assert _main(world.argv("train")) == 2 and not seen
    assert not (world.out_root / "pilot" / "metric").exists()
    monkeypatch.setattr(metric.model, "weights_file_sha256", lambda download=False: WEIGHTS_SHA256)
    assert _main(world.argv("train")) == 0 and seen == [WEIGHTS_SHA256]
    (env,) = (world.out_root / "pilot" / "metric").glob("*/environment.json")
    assert load_json(env)["weights_file_sha256"] == WEIGHTS_SHA256


@pytest.mark.parametrize("case", ["method", "manifest", "base_run", "no_stage"])
def test_search_test_thresholds_provenance(tmp_path, monkeypatch, capsys, case):   # spec 9.5: tD of this model only
    from expcore.io import dump_json
    from expcore.thresholds import Threshold, save_threshold
    from metric.runctx import source_run_id
    from expcore.config import load_config
    world = metric_world(tmp_path, monkeypatch)
    m = load_manifest(world.out_root / "manifests", "v1")
    cfg = load_config(world.cfg_path, module="metric")
    val = world.root / "valstage"
    val.mkdir()
    save_threshold(Threshold("r", "other_method" if case == "method" else "proxy_cal", "c", 0.5, "validation", 3,
                             "f" * 64 if case == "manifest" else m.manifest_hash), val / "thresholds.json")
    if case != "no_stage":
        dump_json({"base_run_id": "other-run" if case == "base_run" else source_run_id(cfg)}, val / "stage.json")
    g = _npz(world, "g.npz", split="test")
    assert _main(world.argv("search", "--gallery", g, "--queries", g, "--pairs",
                            "--thresholds", str(val / "thresholds.json"))) == 2
    assert {"method": "method_id", "manifest": "manifest_hash", "base_run": "base_run_id",
            "no_stage": "stage.json"}[case] in capsys.readouterr().err
    assert not _stage_dirs(world)
