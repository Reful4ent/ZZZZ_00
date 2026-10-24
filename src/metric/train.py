"""Training run (spec 7.1-7.3, 7.6): seeds, init (M-D04), step loop, step log, epoch validation."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from expcore.io import dump_json, write_parquet
from expcore.seeding import init_seeds, stream
from metric.data import (apply_class_subset, apply_per_class_limit, build_proxy_index,
                         check_disjoint, sampler_exclusions, steps_per_epoch)
from metric.errors import RunStop
from metric.model import build_encoder, weights_hash
from metric.pipeline import eval_arrays, train_dataset
from metric.proxies import init_proxies, make_optimizer
from metric.queues import ClassQueues
from metric.sampler import BatchSampler
from metric.search import blocked_topk, no_positive_mask, retrieval_table
from metric.steplog import StepLogger, queue_snapshot, step_stats, write_queue_snapshot
from metric.train_step import StepConfig, make_proxy_step

EPOCH_SCHEMA = {
    "run_id": "str", "epoch": "int64", "global_step": "int64", "val_recall_at_1": "float64",
    "val_recall_at_5": "float64", "val_recall_at_10": "float64", "val_correct_at_1": "int64",
    "n_queries": "int64", "n_valid_queries": "int64", "n_no_positive": "int64",
    "L_base_mean": "float64", "L_dev_mean": "float64", "L_total_mean": "float64",
    "epoch_time_s": "float64", "val_time_s": "float64", "peak_gpu_mem_bytes": "int64",
}
BASE_MINIMAL_NOTE = ("Вариант без очередей измеряет стоимость обучения; его меньшие накладные "
                     "расходы не сравниваются с калибровкой без этого пояснения.")


@dataclass
class TrainContext:
    params: dict
    seed: int
    run_dir: Path
    images: pd.DataFrame
    store: Any
    method: str = "proxy_cal"  # shortcut: method_id until the Method protocol (Task 6.1)
    config_hash: str = ""
    manifest_hash: str = ""
    weights_sha256: str | None = None


def validate_epoch(encoder, method, val_images, store, params) -> dict:
    """Validation Recall@k over the validation split (gallery = same split, self excluded)."""
    inp, s = params["input"], params["search"]
    ids = val_images["image_id"].to_numpy()
    cls = val_images["class_id"].to_numpy()
    bs = params["batch"]["classes"] * params["batch"]["per_class"]
    z = np.concatenate([encoder.embed(tf.constant(x), params["embed"]["eps_z"]).numpy()
                        for _, x in eval_arrays(list(ids), store, inp["resize_short"], inp["crop"],
                                                bs, "none", 0)])
    res = blocked_topk(z, ids, z, ids, max(s["k"]), s["block_q"], s["block_g"], s["d2_tol"],
                       "cosine")
    _, summ = retrieval_table(res, cls, ids, cls, no_positive_mask(ids, cls, ids, cls),
                              ks=tuple(s["k"]))
    return {"val_recall_at_1": summ.get("recall_at_1"), "val_recall_at_5": summ.get("recall_at_5"),
            "val_recall_at_10": summ.get("recall_at_10"), "val_correct_at_1": summ["correct_at_1"],
            "n_queries": summ["n_queries"], "n_valid_queries": summ["n_valid"],
            "n_no_positive": summ["n_no_positive"]}


def export_best(run_dir: Path, encoder) -> str:
    """Minimal best export (weights only); Task 4.2 replaces it with encoder.keras + preprocess.json."""
    d = run_dir / "checkpoints" / "best"
    d.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(d / "encoder.weights.h5"))
    return "checkpoints/best/encoder.weights.h5"


def _check_drift(queues, tol, step):
    if queues is not None:
        drift = queues.max_sum_drift()
        if drift > tol:
            raise RunStop("QUEUE_SUM_DRIFT", {"global_step": step, "max_abs_diff": drift,
                                              "sum_tol": tol})


def _peak_gpu_bytes():
    if not tf.config.list_physical_devices("GPU"):
        return None
    return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])


def _write(df, path, schema=None, nullable=()):
    write_parquet(df, path, schema=schema, nullable=nullable)


def run_training(ctx: TrainContext, resume_state: dict | None = None) -> dict:
    p, seed, run_dir = ctx.params, ctx.seed, Path(ctx.run_dir)
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
    P = init_proxies(n_cls, d, stream(seed, "init", 1))
    queues = ClassQueues(n_cls, cal["K"], d) if cal["queues"] else None
    optimizer = make_optimizer(p["optimizer"]["lr"])
    cfg = StepConfig(kappa=p["proxy"]["kappa"], lam=cal["lambda"], n_warm=cal["n_warm"],
                     eps_z=p["embed"]["eps_z"], eps_p=p["proxy"]["eps_p"])
    step_fn = make_proxy_step(encoder, P, queues, optimizer, cfg)
    sampler = BatchSampler(proxy_index, train, seed, b["classes"], b["per_class"],
                           inp["resize_short"], inp["crop"], inp["train_augment"])

    epochs, check_every, tol = p["train"]["epochs"], p["queue"]["check_every"], p["queue"]["sum_tol"]
    logger = StepLogger(run_dir)
    (run_dir / "queue_state").mkdir(exist_ok=True)
    data = iter(train_dataset(sampler, ctx.store, 0, epochs * spe, prefetch=2))
    batches, epoch_rows, best = [], [], None
    gstep = -1
    for epoch in range(1, epochs + 1):
        t_epoch, losses = time.perf_counter(), []
        for k in range(spe):
            t0 = time.perf_counter()
            x, y, t = next(data)
            t1 = time.perf_counter()
            out = {key: v.numpy() for key, v in step_fn(x, y, t).items()}
            t2 = time.perf_counter()
            gstep = int(t)
            batches.append(sampler.to_frame(sampler.batch(gstep)))
            if queues is None:
                stats = step_stats(P.numpy())
            else:
                mu, n = (v.numpy() for v in queues.centers())
                stats = step_stats(P.numpy(), n, mu, cal["K"], queues.age_stats(gstep)["mean_age"])
            logger.log({
                "run_id": run_id, "global_step": gstep, "epoch": epoch, "step_in_epoch": k,
                "lambda": float(cal["lambda"]), "L_base": float(out["L_base"]),
                "L_dev": float(out["L_dev"]), "L_total": float(out["L_total"]),
                "n_cval": None if queues is None else int(out["n_cval"]), **stats,
                "step_time_ms": (t2 - t1) * 1e3, "data_time_ms": (t1 - t0) * 1e3})
            losses.append((out["L_base"], out["L_dev"], out["L_total"]))
            if (gstep + 1) % check_every == 0:
                _check_drift(queues, tol, gstep)
            # every-N-steps hook point (Task 4.2: checkpoint.every_steps)
        # end of epoch
        _check_drift(queues, tol, gstep)
        logger.end_epoch(epoch)
        write_queue_snapshot(queue_snapshot(queues, P, gstep),
                             run_dir / "queue_state" / f"epoch-{epoch:03d}.parquet")
        _write(pd.concat(batches, ignore_index=True), run_dir / "batches.parquet")
        epoch_time = time.perf_counter() - t_epoch
        t_val = time.perf_counter()
        vrow = validate_epoch(encoder, ctx.method, val, ctx.store, p)
        lm = np.mean(np.asarray(losses, np.float64), axis=0)
        row = {"run_id": run_id, "epoch": epoch, "global_step": gstep, **vrow,
               "L_base_mean": lm[0], "L_dev_mean": lm[1], "L_total_mean": lm[2],
               "epoch_time_s": epoch_time, "val_time_s": time.perf_counter() - t_val,
               "peak_gpu_mem_bytes": _peak_gpu_bytes()}
        epoch_rows.append(row)
        _write(pd.DataFrame(epoch_rows, columns=list(EPOCH_SCHEMA)), run_dir / "epochs.parquet",
               EPOCH_SCHEMA, ("val_recall_at_1", "val_recall_at_5", "val_recall_at_10"))
        if best is None or row["val_correct_at_1"] > best["val_correct_at_1"]:  # tie -> earlier
            best = row
            path = export_best(run_dir, encoder)
            dump_json({
                "epoch": epoch, "global_step": gstep, "val_recall_at_1": row["val_recall_at_1"],
                "val_correct_at_1": row["val_correct_at_1"],
                "n_valid_queries": row["n_valid_queries"],
                "selection_rule": "max_val_recall_at_1_tie_earliest", "encoder_path": path,
                "weights_hash": weights_hash(encoder, ""),
                "disclaimers": [] if queues is not None else [BASE_MINIMAL_NOTE],
            }, run_dir / "best_checkpoint.json")
        # end-of-epoch hook point (Task 4.2: full checkpoint)
    logger.finalize()
    return {"best_epoch": best["epoch"], "best_correct_at_1": best["val_correct_at_1"],
            "total_steps": gstep + 1, "status": "completed", "deterministic_ops": True}
