"""Full checkpoint (spec 7.4), best-encoder export (7.3 item 4) and failure record (12)."""
import shutil
from pathlib import Path

import tensorflow as tf

from expcore.io import dump_json, load_json
from expcore.seeding import capture_rng_state
from metric.model import weights_hash
from metric.steplog import read_steps, write_steps

STATE_KEYS = ("epoch", "global_step", "seed", "sampler_version", "best_epoch",
              "best_correct_at_1", "config_hash", "manifest_hash", "weights_sha256")


def _checkpoint(encoder, P, queues, optimizer, head=()) -> tf.train.Checkpoint:
    """P: proxies or None; head: a method's other extra variables (e.g. classifier kernel/bias)."""
    objs = {"encoder": encoder, "optimizer": optimizer}
    if P is not None:
        objs["P"] = P
    if queues is not None:
        objs["queues"] = queues
    if head:
        objs["head"] = list(head)
    return tf.train.Checkpoint(**objs)


def _sibling(d: Path, suffix: str) -> Path:
    return d.with_name(d.name + suffix)


def _existing(d: Path) -> Path:
    d = Path(d)
    if not d.exists() and _sibling(d, ".old").exists():  # crash between the two swap renames
        return _sibling(d, ".old")
    return d


def save_full(dir: Path, encoder, P, queues, optimizer, state: dict, steps: list = (),
              progress: dict | None = None, head=()) -> None:
    """Write into `<dir>.tmp`, then swap: dir -> .old, .tmp -> dir, drop .old (Windows-safe).

    `steps`/`progress`: the unfinished epoch's step rows and loss accumulators (empty at epoch end),
    saved atomically with the weights so a resumed epoch is complete.
    """
    d = Path(dir)
    tmp, old = _sibling(d, ".tmp"), _sibling(d, ".old")
    if tmp.exists():  # stale tmp of an interrupted save; a failure to remove it must surface
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _checkpoint(encoder, P, queues, optimizer, head).write(str(tmp / "ckpt"))
    write_steps(list(steps), tmp / "partial_steps.parquet")
    dump_json(progress or {}, tmp / "epoch_progress.json")
    dump_json({k: state[k] for k in STATE_KEYS}, tmp / "state.json")
    dump_json(capture_rng_state(), tmp / "rng_state.json")
    if d.exists():
        if old.exists():  # left by a cleanup that failed (e.g. a file handle held on Windows)
            shutil.rmtree(old)
        d.rename(old)
    tmp.rename(d)
    shutil.rmtree(old, ignore_errors=True)


def restore_full(dir: Path, encoder, P, queues, optimizer, head=()) -> dict:
    """Restore all §7.4 state in place (optimizer must already be built); returns state.json."""
    d = _existing(dir)
    _checkpoint(encoder, P, queues, optimizer, head).read(str(d / "ckpt")).assert_consumed()
    return read_state(d)


def read_state(dir: Path) -> dict:
    return load_json(_existing(dir) / "state.json")


def has_state(dir: Path) -> bool:
    """A full checkpoint (or its .old sibling left by an interrupted swap) exists."""
    return (_existing(dir) / "state.json").is_file()


def load_progress(dir: Path) -> tuple[list[dict], dict]:
    """(step rows, accumulators) of the unfinished epoch saved by save_full, to seed StepLogger.rows."""
    d = _existing(dir)
    return read_steps(d / "partial_steps.parquet"), load_json(d / "epoch_progress.json")


def export_best(dir: Path, encoder, preprocess: dict) -> str:
    """encoder.keras (inner functional backbone -> dense; L2 normalisation per preprocess.json)."""
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    inp = tf.keras.Input(encoder.backbone.input_shape[1:])
    # the subclassed Encoder has no get_config, so Keras 3 cannot save it as .keras
    model = tf.keras.Model(inp, encoder.dense(encoder.backbone(inp, training=False)))
    model.save(d / "encoder.keras")
    dump_json(preprocess, d / "preprocess.json")
    return weights_hash(encoder, "")


def write_failure(run_dir, failure_type: str, details: dict) -> None:
    dump_json({"failure_type": failure_type, "details": details}, Path(run_dir) / "failure.json")
