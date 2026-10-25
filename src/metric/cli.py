"""python -m metric train|resume|embed|search|select (spec 13.1). TensorFlow is imported only inside handlers.

Run-dir layout (out_root/<profile>/metric/):
  <run_id of config C>/                  train (base run of C): checkpoints, logs, best_checkpoint.json, memory.json
  <run_id of stage hash>/                embed / search runs, stage.json names the stage and the train run
  select/<experiments config stem>/selected_config.json
"""
import argparse
import time
from pathlib import Path
from typing import Sequence

from expcore.cli_common import DEFAULT_REGISTRY, add_common_args, path_overrides, run_cli
from expcore.config import load_config
from expcore.errors import ConfigError, ProtocolError, ProtocolViolationError, TechnicalError
from expcore.hashing import canonical_json, sha256_file
from expcore.io import atomic_write_bytes, dump_json, load_json
from expcore.resume import SUCCESS
from expcore.runlog import JsonlLogger, now_utc
from expcore.timing import peak_memory_mb
import metric.params  # noqa: F401  (registers the params schema)
from metric.errors import RunStop

MODULE = "metric"
UNTRAINABLE = ("imagenet_fixed", "rgb_hist")   # = metric.methods.UNTRAINABLE (methods imports TF)
RAW_L2_METHODS = ("proxy_softmax_noL2",)       # search_metric "raw_l2" (metric.methods.ProxyMethod)


def _enable_tf_determinism() -> bool | None:
    """True/False if TensorFlow is installed (op determinism available or not), None without TensorFlow."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    enable = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable is None:
        return False
    enable()
    return True


# --- run lifecycle -----------------------------------------------------------------------------

def _write_timers(run, started: str, t0: float, status: str) -> None:
    path = run.run_dir / "timers.json"
    timers = load_json(path) if path.exists() else {"attempts": []}
    peak, reason = peak_memory_mb()
    timers["attempts"].append({"started_at": started, "status": status,
                               "total_ms": (time.perf_counter() - t0) * 1e3, "peak_memory_mb": peak,
                               "peak_memory_reason": reason})
    dump_json(timers, path)


def _fail(run, ftype: str, message: str) -> None:
    failure = {"type": ftype, "message": message, "unit_id": None}
    for name in ("errors.jsonl", "run.log.jsonl"):
        JsonlLogger(run.run_dir / name, run.run_id).log("ERROR", "run_failed", **failure)
    run.record.finish("failed", failure)


def _execute(run, body) -> None:
    """start → body(run) → _SUCCESS + completed; any exception → failed run.json, errors.jsonl, timers.json."""
    run.record.start()
    run.write_run_files()
    started, t0 = now_utc(), time.perf_counter()
    try:
        body(run)
    except Exception as e:   # KeyboardInterrupt passes: the attempt stays "running"
        _write_timers(run, started, t0, "failed")
        if isinstance(e, RunStop):
            _fail(run, e.failure_type, canonical_json(e.details))
            raise
        _fail(run, type(e).__name__ if isinstance(e, ProtocolError) else "TechnicalError", str(e))
        if isinstance(e, ProtocolError):
            raise
        raise TechnicalError(f"{type(e).__name__}: {e}") from e
    _write_timers(run, started, t0, "completed")
    atomic_write_bytes(run.run_dir / SUCCESS, b"")
    run.record.finish("completed")
    JsonlLogger(run.run_dir / "run.log.jsonl", run.run_id).log("INFO", "run_completed")


def _completed(run) -> bool:
    run_json = run.run_dir / "run.json"
    if (run.run_dir / SUCCESS).exists() and run_json.exists() and load_json(run_json)["status"] == "completed":
        print(f"{run.run_dir}: completed, not recomputed")
        return True
    return False


# --- train / resume ----------------------------------------------------------------------------

def _train(args: argparse.Namespace, checkpoint: Path | None = None) -> None:
    from metric.runctx import prepare_run
    deterministic = _enable_tf_determinism()   # before init_seeds, so TF gets seeded too
    run = prepare_run("train", args, deterministic_ops=deterministic)
    if run.method_id in UNTRAINABLE:
        raise ConfigError(f"{run.method_id} has no training: run `metric embed` for it")
    auto = False
    if checkpoint is not None:
        if checkpoint.resolve().parent.parent != run.run_dir.resolve():
            raise ConfigError(f"{checkpoint}: not a checkpoint of {run.run_dir}")
    elif _completed(run):
        return
    else:   # a failed/interrupted attempt: continue from its checkpoint (M-D03: same result as from scratch)
        from metric.checkpoint import has_state
        last = run.run_dir / "checkpoints" / "last"
        if has_state(last):
            checkpoint, auto = last, True
    from metric import train as T
    p = run.params
    ctx = T.TrainContext(params=p, seed=run.cfg.seeds.seed, run_dir=run.run_dir, images=run.images, store=run.store,
                         method=run.method_id, config_hash=run.config_hash,
                         manifest_hash=run.manifest.manifest_hash, weights_sha256=run.weights_sha256)

    def body(run):
        log = JsonlLogger(run.run_dir / "run.log.jsonl", run.run_id)
        if auto:
            from metric.checkpoint import read_state
            log.log("INFO", "auto_resume", checkpoint="checkpoints/last",
                    global_step=read_state(checkpoint)["global_step"])
        res = T.run_training(ctx) if checkpoint is None else T.resume_training(ctx, checkpoint)
        for check in res.get("queue_checks", []):
            log.log("INFO", "queue_check", **check)
        if res.get("memory") is not None:
            dump_json(res["memory"], run.run_dir / "memory.json")
        if res["status"] == "failed":
            raise RunStop(res["failure"]["type"], res["failure"]["details"])
        # a completed attempt drops the earlier attempt's failure.json (errors.jsonl and run.json keep the history)
        (run.run_dir / "failure.json").unlink(missing_ok=True)
    _execute(run, body)


def _resume(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    args.config = checkpoint.parent.parent / "config.resolved.yaml"   # spec 7.5: the run's own config
    if not args.config.is_file():
        raise ConfigError(f"{args.config}: missing; {checkpoint} is not a checkpoint of a metric train run")
    _train(args, checkpoint)


# --- embed -------------------------------------------------------------------------------------

class _LoadedEncoder:
    """The exported best encoder (backbone → dense, pre-L2) behind the Encoder interface Embedder uses."""

    def __init__(self, model):
        self.model, self.variables = model, model.variables

    def embed_raw(self, x):
        return self.model(x, training=False)


def _embed(args: argparse.Namespace) -> None:
    """Trained methods: a stage run per --split. Untrainable methods without --split: the base run of the config
    (as `experiments plan` names it) with embeddings-<split>.npz for every split the profile allows."""
    from expcore.loaders import accessible_splits
    from metric.runctx import completed_source, prepare_run
    deterministic = _enable_tf_determinism()
    cfg = load_config(args.config, module=MODULE, path_overrides=path_overrides(args))
    trained = next(iter(cfg.algorithms), None) not in UNTRAINABLE
    if args.split is None and trained:
        raise ConfigError("embed --split is required for a trained model")
    stage = None if args.split is None else {"command": "embed", "split": args.split,
                                             "degradation_id": cfg.params.degradation}
    run = prepare_run("embed", args, deterministic_ops=deterministic, stage=stage)
    if _completed(run):
        return
    source = completed_source(run.cfg) if trained else None
    if args.split is None:
        fixture = all(ds.fixture for ds in run.cfg.datasets.values())
        splits = {s: run.run_dir / f"embeddings-{s}.npz"
                  for s in accessible_splits(run.cfg.profile, fixture=fixture, splits={"validation", "test"})}
    else:
        splits = {args.split: run.run_dir / "embeddings.npz"}

    def body(run):
        for split in splits:
            run.open_split(split)   # after record.start(), before any image of the split is loaded
        import pandas as pd
        import tensorflow as tf
        from metric.embed import embed_split, save_embeddings
        from metric.methods import build_method, make_embedder
        p, seed = run.params, run.cfg.seeds.seed
        if trained:
            n_cls = len(pd.read_parquet(source / "proxy_index.parquet"))
            method = build_method(run.method_id, p, n_cls, seed)
            encoder = _LoadedEncoder(tf.keras.models.load_model(source / "checkpoints" / "best" / "encoder.keras"))
        else:
            method, encoder = build_method(run.method_id, p, 0, seed), None
        for split, out in splits.items():
            es = embed_split(make_embedder(method, encoder, p), run.images, split, run.store, p,
                             p["degradation"], seed, run.run_dir / "embeddings", run.manifest.manifest_hash)
            save_embeddings(out, es)
    _execute(run, body)


# --- search ------------------------------------------------------------------------------------

def _write_predictions(run, rows, expected) -> None:
    """One shard in expcore's layout (predictions/part-000000.parquet + progress.json), read by read_predictions."""
    from expcore.io import write_parquet
    from expcore.schemas import PREDICTIONS_COLUMNS, PREDICTIONS_NULLABLE, validate_predictions
    validate_predictions(rows, expected, require_complete=True)
    shard = run.run_dir / "predictions" / "part-000000.parquet"
    shard.parent.mkdir(exist_ok=True)
    write_parquet(rows[list(PREDICTIONS_COLUMNS)], shard, schema=PREDICTIONS_COLUMNS, nullable=PREDICTIONS_NULLABLE)
    dump_json({"units": {"0": {"unit_id": 0, "n_rows": len(rows), "sha256": sha256_file(shard)}}},
              run.run_dir / "progress.json")


def _check_threshold_provenance(run, path: Path) -> None:
    """tD must come from this method, this manifest and the validation search of the same base training run."""
    from expcore.thresholds import load_threshold
    from metric.runctx import source_run_id
    t = load_threshold(path)
    if t.method_id != run.method_id:
        raise ConfigError(f"{path}: threshold method_id {t.method_id!r} != {run.method_id!r}")
    if t.manifest_hash != run.manifest.manifest_hash:
        raise ConfigError(f"{path}: threshold manifest_hash differs from this config's manifest")
    stage = path.parent / "stage.json"
    if not stage.is_file():
        raise ConfigError(f"{stage}: missing; the threshold's base training run cannot be verified")
    base = load_json(stage).get("base_run_id")
    if base != source_run_id(run.cfg):
        raise ConfigError(f"{path}: threshold base_run_id {base!r} != this config's {source_run_id(run.cfg)!r}")


def _search(args: argparse.Namespace) -> None:
    import pandas as pd
    from expcore.thresholds import load_threshold
    from metric.embed import load_embeddings
    from metric.evaluate import evaluate_pairs, evaluate_retrieval, pair_prediction_rows, query_prediction_rows
    from metric.pairs import pairs_summary
    from metric.runctx import prepare_run
    gallery, queries = load_embeddings(args.gallery), load_embeddings(args.queries)
    split = queries.split
    if args.pairs and queries.degradation_id != "none":   # spec 9.7: degradations apply to retrieval queries only
        raise ConfigError(f"search --pairs: queries are degraded ({queries.degradation_id!r}); pairs and tD use "
                          "undegraded embeddings")
    if args.pairs and split != "validation" and not args.thresholds:
        raise ProtocolViolationError(f"search --pairs on {split!r} needs --thresholds <validation thresholds.json>")
    stage = {"command": "search", "split": split, "degradation_id": queries.degradation_id,
             "gallery_sha256": sha256_file(args.gallery), "queries_sha256": sha256_file(args.queries),
             "pairs": bool(args.pairs),
             "thresholds_sha256": sha256_file(args.thresholds) if args.thresholds else None}
    run = prepare_run("search", args, deterministic_ops=None, stage=stage)
    if _completed(run):
        return
    if args.thresholds:
        _check_threshold_provenance(run, Path(args.thresholds))
    if args.pairs and run.method_id in RAW_L2_METHODS:
        raise ConfigError(f"search --pairs: pair scores are cosine (spec 9.6); {run.method_id} searches raw L2")
    for name, es in (("gallery", gallery), ("queries", queries)):
        if es.method_id != run.method_id or es.manifest_hash != run.manifest.manifest_hash:
            raise ConfigError(f"--{name}: embeddings of {es.method_id!r} / manifest {es.manifest_hash[:12]}, "
                              f"this config is {run.method_id!r} / {run.manifest.manifest_hash[:12]}")
    if gallery.weights_hash != queries.weights_hash:
        raise ConfigError("--gallery and --queries come from different encoders (weights_hash differs)")
    if gallery.split != split or gallery.degradation_id != "none":
        raise ConfigError(f"--gallery: an undegraded gallery of split {split!r} is required (spec 4.4, 9.7)")

    def body(run):
        run.open_split(split)   # after record.start(): test is never opened without a run attempt
        p = run.params
        p_eval = {**p, "search": {**p["search"], "k": max(p["search"]["k"])}}
        ids = {"out_dir": run.run_dir, "run_id": run.run_id, "method_id": run.method_id,
               "manifest_hash": run.manifest.manifest_hash}
        td = None
        if args.thresholds:
            td = -load_threshold(Path(args.thresholds)).value   # unified score s = -d (metric.evaluate)
        t0 = time.perf_counter()
        sc = pairs = None
        if args.pairs:
            pairs = run.manifest.pairs["p3"]
            pairs = pairs[pairs["split"] == split].sort_values("pair_id", kind="mergesort").reset_index(drop=True)
            own = run.run_dir / "thresholds.json"
            path = own if split == "validation" else (Path(args.thresholds) if args.thresholds else None)
            sc = evaluate_pairs(queries, pairs, split, p, path, ids=ids)
            dump_json(pairs_summary(pairs, run.images), run.run_dir / "p3_pairs_summary.json")   # M-D30
            if split == "validation":
                td = -load_threshold(own).value
        classes = run.images.set_index("image_id")["class_id"]
        expected = sorted(run.images.loc[run.images["split"] == split, "image_id"])
        metric = "raw_l2" if run.method_id in RAW_L2_METHODS else "cosine"
        df, _ = evaluate_retrieval(gallery, queries, classes, p_eval, ids, expected_gallery_ids=expected, td=td,
                                   metric=metric)
        elapsed = (time.perf_counter() - t0) * 1e3
        if sc is not None:   # Task 5.2 ruling: pair rows with --pairs, query rows otherwise (TZ00 INV-07)
            rows = pair_prediction_rows(sc, td, run_id=run.run_id, method_id=run.method_id, elapsed_ms=elapsed)
            _write_predictions(run, rows, pairs)
        elif td is not None:
            rows = query_prediction_rows(df, td, run_id=run.run_id, method_id=run.method_id, elapsed_ms=elapsed)
            _write_predictions(run, rows, pd.DataFrame({"query_id": df["query_id"]}))
        else:   # decided query rows need y_pred, i.e. tD (INV-05); retrieval.parquet holds the results
            JsonlLogger(run.run_dir / "run.log.jsonl", run.run_id).log(
                "INFO", "predictions_skipped", reason="no threshold: pass --pairs on validation or --thresholds")
    _execute(run, body)


# --- select ------------------------------------------------------------------------------------

def _select(args: argparse.Namespace) -> None:
    from metric.select import select
    out = select(args.config, path_overrides(args))
    print(out)


# --- entry -------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=MODULE)
    sub = parser.add_subparsers(dest="command", required=True)
    add_common_args(sub.add_parser("train", help="train a config (TZ00 run)"))
    r = sub.add_parser("resume", help="continue a train run from its full checkpoint (spec 7.5)")
    r.add_argument("--checkpoint", type=Path, required=True, help="<run_dir>/checkpoints/last")
    r.add_argument("--data-root", type=Path)
    r.add_argument("--manifest-dir", type=Path)
    r.add_argument("--out-root", type=Path)
    r.add_argument("--protocol-registry", type=Path, default=DEFAULT_REGISTRY)
    e = add_common_args(sub.add_parser("embed", help="embed a split with the best encoder of the train run"))
    e.add_argument("--split", choices=("validation", "test"),
                   help="required for trained models; untrainable methods without it embed every allowed split")
    s = add_common_args(sub.add_parser("search", help="blocked search, Recall@k; --pairs: pair scores, tD"))
    s.add_argument("--gallery", type=Path, required=True, help="embeddings.npz of the gallery split")
    s.add_argument("--queries", type=Path, required=True, help="embeddings.npz of the queries")
    s.add_argument("--pairs", action="store_true", help="score the p3 pairs of the split (validation selects tD)")
    s.add_argument("--thresholds", type=Path, help="validation thresholds.json (required for test with --pairs)")
    add_common_args(sub.add_parser("select", help="selected_config.json over tuning runs (--config: experiments "
                                                  "config of the tuning grid)"))
    return parser


HANDLERS = {"train": _train, "resume": _resume, "embed": _embed, "search": _search, "select": _select}


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(lambda a: HANDLERS[a.command](a), argv, parser=_parser())
