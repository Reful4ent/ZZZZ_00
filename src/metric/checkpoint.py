"""Full checkpoint (spec 7.4), best-encoder export (7.3 item 4) and failure record (12)."""
import shutil
from pathlib import Path

import tensorflow as tf

from expcore.io import dump_json, load_json
from expcore.seeding import capture_rng_state
from metric.model import weights_hash

STATE_KEYS = ("epoch", "global_step", "seed", "sampler_version", "best_epoch",
              "best_correct_at_1", "config_hash", "manifest_hash", "weights_sha256")


def _checkpoint(encoder, P, queues, optimizer) -> tf.train.Checkpoint:
    objs = {"encoder": encoder, "P": P, "optimizer": optimizer}
    if queues is not None:
        objs["queues"] = queues
    return tf.train.Checkpoint(**objs)


def _sibling(d: Path, suffix: str) -> Path:
    return d.with_name(d.name + suffix)


def save_full(dir: Path, encoder, P, queues, optimizer, state: dict) -> None:
    """Write into `<dir>.tmp`, then swap: dir -> .old, .tmp -> dir, drop .old (Windows-safe)."""
    d = Path(dir)
    tmp, old = _sibling(d, ".tmp"), _sibling(d, ".old")
    shutil.rmtree(tmp, ignore_errors=True)  # stale tmp of an interrupted save
    tmp.mkdir(parents=True)
    _checkpoint(encoder, P, queues, optimizer).write(str(tmp / "ckpt"))
    dump_json({k: state[k] for k in STATE_KEYS}, tmp / "state.json")
    dump_json(capture_rng_state(), tmp / "rng_state.json")
    if d.exists():
        shutil.rmtree(old, ignore_errors=True)
        d.rename(old)
    tmp.rename(d)
    shutil.rmtree(old, ignore_errors=True)


def restore_full(dir: Path, encoder, P, queues, optimizer) -> dict:
    """Restore all §7.4 state in place (optimizer must already be built); returns state.json."""
    d = Path(dir)
    if not d.exists() and _sibling(d, ".old").exists():  # crash between the two swap renames
        d = _sibling(d, ".old")
    _checkpoint(encoder, P, queues, optimizer).read(str(d / "ckpt")).assert_consumed()
    return load_json(d / "state.json")


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
