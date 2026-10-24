"""Training run (spec 7.1-7.6): seeds, init (M-D04), step loop, step log, epoch validation, resume."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from expcore.errors import ConfigError
from expcore.io import dump_json, write_parquet
from expcore.seeding import init_seeds, stream
from metric.checkpoint import (export_best, load_progress, read_state, restore_full, save_full,
                               write_failure)
from metric.data import (apply_class_subset, apply_per_class_limit, build_proxy_index,
                         check_disjoint, sampler_exclusions, steps_per_epoch)
from metric.errors import RunStop
from metric.methods import PROXY_STEP, UNTRAINABLE, build_method, make_embedder
from metric.model import build_encoder
from metric.pipeline import eval_arrays, train_dataset
from metric.proxies import make_optimizer
from metric.queues import ClassQueues
from metric.sampler import SAMPLER_VERSION, BatchSampler
from metric.search import blocked_topk, no_positive_mask, retrieval_table
from metric.steplog import StepLogger, queue_snapshot, step_stats, write_queue_snapshot
from metric.train_step import StepConfig, make_method_step, make_proxy_step

EPOCH_SCHEMA = {
    "run_id": "str", "epoch": "int64", "global_step": "int64", "val_recall_at_1": "float64",
    "val_recall_at_5": "float64", "val_recall_at_10": "float64", "val_correct_at_1": "int64",
    "n_queries": "int64", "n_valid_queries": "int64", "n_no_positive": "int64",
    "L_base_mean": "float64", "L_dev_mean": "float64", "L_total_mean": "float64",
    "epoch_time_s": "float64", "val_time_s": "float64", "peak_gpu_mem_bytes": "int64",
}
# graph-side stops: tf.debugging.Assert messages start with the failure type (metric.losses)
GRAPH_FAILURES = ("INVALID_EMBEDDING_NORM", "INVALID_PROXY_NORM", "SAMPLER_ERROR",
                  "DISTANCE_RANGE_VIOLATION")
BASE_MINIMAL_NOTE = ("Вариант без очередей измеряет стоимость обучения; его меньшие накладные "
                     "расходы не сравниваются с калибровкой без этого пояснения.")


@dataclass
class TrainContext:
    params: dict
    seed: int
    run_dir: Path
    images: pd.DataFrame
    store: Any
    method: str = "proxy_cal"  # method_id (`algorithms` key); the Method is built by run_training
    config_hash: str = ""
    manifest_hash: str = ""
    weights_sha256: str | None = None


def validate_epoch(encoder, method, val_images, store, params) -> dict:
    """Validation Recall@k over the validation split (gallery = same split, self excluded).

    Embeds with the method's search embedding and searches with its metric.
    """
    inp, s = params["input"], params["search"]
    ids = val_images["image_id"].to_numpy()
    cls = val_images["class_id"].to_numpy()
    bs = params["batch"]["classes"] * params["batch"]["per_class"]
    emb = make_embedder(method, encoder, params)
    z = np.concatenate([emb.embed_images(x) for _, x in eval_arrays(
        list(ids), store, inp["resize_short"], inp["crop"], bs, "none", 0)])
    res = blocked_topk(z, ids, z, ids, max(s["k"]), s["block_q"], s["block_g"], s["d2_tol"],
                       method.search_metric)
    _, summ = retrieval_table(res, cls, ids, cls, no_positive_mask(ids, cls, ids, cls),
                              ks=tuple(s["k"]))
    return {"val_recall_at_1": summ.get("recall_at_1"), "val_recall_at_5": summ.get("recall_at_5"),
            "val_recall_at_10": summ.get("recall_at_10"), "val_correct_at_1": summ["correct_at_1"],
            "n_queries": summ["n_queries"], "n_valid_queries": summ["n_valid"],
            "n_no_positive": summ["n_no_positive"]}


def _check_drift(queues, tol, step) -> dict | None:
    if queues is None:
        return None
    check = {"global_step": step, "max_abs_diff": queues.max_sum_drift(), "sum_tol": tol}
    if check["max_abs_diff"] > tol:
        raise RunStop("QUEUE_SUM_DRIFT", check)
    return check


def _failure(e, cur, gstep, sampler) -> tuple[str, dict]:
    """(failure_type, details) of a run stop; `cur` is the step being executed, if any."""
    step = gstep if cur is None else cur
    if isinstance(e, RunStop):
        return e.failure_type, e.details
    if isinstance(e, tf.errors.ResourceExhaustedError):
        return "OOM", {"global_step": step, "peak_gpu_mem_bytes": _peak_gpu_bytes()}
    ftype = next((t for t in GRAPH_FAILURES if t in e.message), None)
    if ftype is None:
        raise e
    details = {"global_step": step}
    if cur is not None:
        details["image_ids"] = sampler.batch(cur).image_ids
    return ftype, details


def _peak_gpu_bytes():
    if not tf.config.list_physical_devices("GPU"):
        return None
    return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])


def _write(df, path, schema=None, nullable=()):
    write_parquet(df, path, schema=schema, nullable=nullable)


def _prior(path: Path, col: str, upto: int) -> pd.DataFrame:
    """Rows of an earlier attempt's table with `col < upto` (a later partial write is dropped)."""
    df = pd.read_parquet(path) if path.exists() else pd.DataFrame({col: []})
    return df[df[col] < upto]


def run_training(ctx: TrainContext) -> dict:
    return _run(ctx)


def resume_training(ctx: TrainContext, checkpoint_dir: Path) -> dict:
    """Continue from the full checkpoint at global_step + 1 (spec 7.5); ctx is the run's config."""
    st = read_state(checkpoint_dir)
    expected = {"config_hash": ctx.config_hash, "manifest_hash": ctx.manifest_hash,
                "weights_sha256": ctx.weights_sha256, "seed": ctx.seed,
                "sampler_version": SAMPLER_VERSION}  # batches depend on (seed, sampler version)
    for k, v in expected.items():
        if st[k] != v:
            raise ConfigError(f"resume: {k} {v!r} != checkpoint {st[k]!r}")
    return _run(ctx, Path(checkpoint_dir))


def _run(ctx: TrainContext, ckpt: Path | None = None) -> dict:
    p, seed, run_dir = ctx.params, ctx.seed, Path(ctx.run_dir)
    if ctx.method in UNTRAINABLE:
        raise ConfigError(f"{ctx.method} has no training: embed it directly")
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    check_disjoint(ctx.images)
    train = ctx.images[ctx.images["split"] == "train"].reset_index(drop=True)
    val = ctx.images[ctx.images["split"] == "validation"].reset_index(drop=True)
    dp = p["data"]
    if dp["class_subset"] is not None:
        train = apply_class_subset(train, dp["class_subset"], seed)
    if dp["per_class_limit"] is not None or dp["cohort_min_images"] is not None:
        train = apply_per_class_limit(train, dp["per_class_limit"], dp["cohort_min_images"] or 0)
    proxy_index = build_proxy_index(train)
    _write(proxy_index, run_dir / "proxy_index.parquet")
    _write(sampler_exclusions(proxy_index), run_dir / "sampler_exclusions.parquet")

    b, inp, cal = p["batch"], p["input"], p["calibration"]
    bs = b["classes"] * b["per_class"]
    spe = p["train"]["steps_per_epoch"]
    if spe == "auto":
        elig = set(proxy_index.loc[proxy_index["sampler_eligible"], "class_id"])
        spe = steps_per_epoch(int(train["class_id"].isin(elig).sum()), bs)

    init_seeds(seed)  # tf is imported, so this also seeds TF/Keras
    tf.config.experimental.enable_op_determinism()
    n_cls, d = len(proxy_index), p["model"]["embedding_dim"]
    dense_seed = int(stream(seed, "init", 0).integers(2**31 - 1))
    encoder = build_encoder(p["model"]["weights"], inp["crop"], d, dense_seed)
    method = build_method(ctx.method, p, n_cls, seed)
    P = getattr(method, "P", None)
    head = [v for v in method.extra_variables if v is not P]  # checkpointed beside P
    queues = ClassQueues(n_cls, cal["K"], d) if method.uses_queues else None
    optimizer = make_optimizer(p["optimizer"]["lr"])
    cfg = StepConfig(kappa=p["proxy"]["kappa"], lam=cal["lambda"], n_warm=cal["n_warm"],
                     eps_z=p["embed"]["eps_z"], eps_p=p["proxy"]["eps_p"])
    proxy_step = ctx.method in PROXY_STEP
    if proxy_step:
        step_fn = make_proxy_step(encoder, P, queues, optimizer, cfg)
    else:
        step_fn = make_method_step(encoder, method, optimizer, cfg)
    epochs, check_every, tol = p["train"]["epochs"], p["queue"]["check_every"], p["queue"]["sum_tol"]
    every = p["checkpoint"]["every_steps"]
    preprocess = {"resize_short": inp["resize_short"], "crop": inp["crop"], "channel_order": "RGB",
                  "preprocess": "tf.keras.applications.resnet50.preprocess_input",
                  "embedding_dim": d, "normalize_l2": inp["normalize_l2"]}
    logger = StepLogger(run_dir)
    (run_dir / "queue_state").mkdir(exist_ok=True)
    batch_frames, epoch_batches, epoch_rows, queue_checks = [], [], [], []
    best, sampler, cur, epoch, gstep, losses = None, None, None, 0, -1, []
    if ckpt is not None:  # batches are pure functions of (seed, step): only the state is restored
        st = restore_full(ckpt, encoder, P, queues, optimizer, head)
        gstep = st["global_step"]
        if st["best_epoch"] is not None:
            best = {"epoch": st["best_epoch"], "val_correct_at_1": st["best_correct_at_1"]}
        logger.rows, prog = load_progress(ckpt)
        losses = prog["losses"]
    start = gstep + 1
    epoch0, k0 = start // spe + 1, start % spe  # the unfinished epoch is replayed from its rows
    epoch = epoch0  # a stop before the loop files seeded rows under this epoch
    if ckpt is not None:
        ep = _prior(run_dir / "epochs.parquet", "epoch", epoch0)
        epoch_rows = ep.astype(object).where(ep.notna(), None).to_dict("records")
        prev = _prior(run_dir / "batches.parquet", "global_step", start - k0)
        batch_frames = [prev] if len(prev) else []

    def check(step):
        c = _check_drift(queues, tol, step)
        if c is not None:
            queue_checks.append(c)  # shortcut: journal write is Task 7.1

    def best_fields():
        return {"best_epoch": best and best["epoch"],
                "best_correct_at_1": best and best["val_correct_at_1"]}

    def save():
        save_full(run_dir / "checkpoints" / "last", encoder, P, queues, optimizer, {
            "epoch": epoch, "global_step": gstep, "seed": seed, "sampler_version": SAMPLER_VERSION,
            **best_fields(), "config_hash": ctx.config_hash, "manifest_hash": ctx.manifest_hash,
            "weights_sha256": ctx.weights_sha256},
            steps=logger.rows, progress={"losses": np.asarray(losses, np.float64).tolist()},
            head=head)

    try:
        sampler = BatchSampler(proxy_index, train, seed, b["classes"], b["per_class"],
                               inp["resize_short"], inp["crop"], inp["train_augment"])
        epoch_batches = [sampler.to_frame(sampler.batch(s)) for s in range(start - k0, start)]
        data = iter(train_dataset(sampler, ctx.store, start, epochs * spe - start, prefetch=2))
        for epoch in range(epoch0, epochs + 1):
            t_epoch = time.perf_counter()
            for k in range(k0 if epoch == epoch0 else 0, spe):
                t0 = time.perf_counter()
                x, y, t = next(data)
                t1 = time.perf_counter()
                cur = int(t)
                out = {key: v.numpy() for key, v in step_fn(x, y, t).items()}
                t2 = time.perf_counter()
                gstep, cur = cur, None
                epoch_batches.append(sampler.to_frame(sampler.batch(gstep)))
                if queues is None:
                    stats = step_stats(None if P is None else P.numpy())
                else:
                    mu, n = (v.numpy() for v in queues.centers())
                    stats = step_stats(P.numpy(), n, mu, cal["K"],
                                       queues.age_stats(gstep)["mean_age"])
                logger.log({
                    "run_id": run_id, "global_step": gstep, "epoch": epoch, "step_in_epoch": k,
                    "lambda": float(cal["lambda"]), "L_base": float(out["L_base"]),
                    "L_dev": float(out["L_dev"]), "L_total": float(out["L_total"]),
                    "n_cval": None if queues is None else int(out["n_cval"]), **stats,
                    "step_time_ms": (t2 - t1) * 1e3, "data_time_ms": (t1 - t0) * 1e3})
                losses.append((out["L_base"], out["L_dev"], out["L_total"]))
                if (gstep + 1) % check_every == 0:
                    check(gstep)
                if every and (gstep + 1) % every == 0 and k < spe - 1:  # epoch end saves anyway
                    check(gstep)
                    save()
            # end of epoch
            check(gstep)
            logger.end_epoch(epoch)
            if P is not None:
                write_queue_snapshot(queue_snapshot(queues, P, gstep),
                                     run_dir / "queue_state" / f"epoch-{epoch:03d}.parquet")
            batch_frames.append(pd.concat(epoch_batches, ignore_index=True))
            epoch_batches = []
            _write(pd.concat(batch_frames, ignore_index=True), run_dir / "batches.parquet")
            epoch_time = time.perf_counter() - t_epoch
            t_val = time.perf_counter()
            vrow = validate_epoch(encoder, method, val, ctx.store, p)
            lm = np.mean(np.asarray(losses, np.float64), axis=0)
            losses = []  # epoch flushed: the epoch-end checkpoint carries no partial progress
            row = {"run_id": run_id, "epoch": epoch, "global_step": gstep, **vrow,
                   "L_base_mean": lm[0], "L_dev_mean": lm[1], "L_total_mean": lm[2],
                   "epoch_time_s": epoch_time, "val_time_s": time.perf_counter() - t_val,
                   "peak_gpu_mem_bytes": _peak_gpu_bytes()}
            epoch_rows.append(row)
            _write(pd.DataFrame(epoch_rows, columns=list(EPOCH_SCHEMA)),
                   run_dir / "epochs.parquet", EPOCH_SCHEMA,
                   ("val_recall_at_1", "val_recall_at_5", "val_recall_at_10"))
            if best is None or row["val_correct_at_1"] > best["val_correct_at_1"]:  # tie->earlier
                best = row
                whash = export_best(run_dir / "checkpoints" / "best", encoder, preprocess)
                dump_json({
                    "epoch": epoch, "global_step": gstep,
                    "val_recall_at_1": row["val_recall_at_1"],
                    "val_correct_at_1": row["val_correct_at_1"],
                    "n_valid_queries": row["n_valid_queries"],
                    "selection_rule": "max_val_recall_at_1_tie_earliest",
                    "encoder_path": "checkpoints/best/encoder.keras", "weights_hash": whash,
                    "disclaimers": [BASE_MINIMAL_NOTE] if proxy_step and queues is None else [],
                }, run_dir / "best_checkpoint.json")
            save()
    except (RunStop, tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError) as e:
        ftype, details = _failure(e, cur, gstep, sampler)
        # keep the partial epoch's rows; `last` stays the pre-failure checkpoint
        if logger.rows:
            logger.end_epoch(epoch)
        if epoch_batches:
            _write(pd.concat([*batch_frames, *epoch_batches], ignore_index=True),
                   run_dir / "batches.parquet")
        if any(logger.parts.glob("part-*.parquet")):
            logger.finalize()
        write_failure(run_dir, ftype, details)
        result = {"status": "failed", "failure": {"type": ftype, "details": details}}
    else:
        logger.finalize()
        result = {"status": "completed"}
    return {**result, **best_fields(), "total_steps": gstep + 1, "deterministic_ops": True,
            "queue_checks": queue_checks}
