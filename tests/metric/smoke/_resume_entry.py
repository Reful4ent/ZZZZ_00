"""Subprocess entry for ПРИЁМ-М3: `python -m tests.metric.smoke._resume_entry run_dir mode [variant]`.

mode: train (uninterrupted) | train-5 (process dies after 5 steps, like a kill) | resume.
variant: smoke (56 px, weights=None) | resnet (224 px, weights=imagenet, 16 classes).
"""
import sys
from pathlib import Path

from tests.metric.smoke.fixtures import make_sop_like, smoke_params

KILL_AFTER = 5


class _Killed(Exception):
    """Not caught by run_training: the process stops as if killed, after step KILL_AFTER's save."""


def params(variant):
    p = smoke_params(lam=0.1, epochs=4, steps_per_epoch=3)  # step 5 (gstep 4) is mid-epoch 2
    p["checkpoint"]["every_steps"] = 5
    if variant == "resnet":
        p["model"]["weights"] = "imagenet"
        p["input"].update(resize_short=256, crop=224)
    return p


def context(run_dir, variant="smoke", **hashes):
    import metric.train as train
    images, store = make_sop_like(n_classes=16) if variant == "resnet" else make_sop_like()
    return train.TrainContext(params=params(variant), seed=7, run_dir=Path(run_dir), images=images,
                              store=store, config_hash=hashes.get("config_hash", "c"),
                              manifest_hash=hashes.get("manifest_hash", "m"),
                              weights_sha256=hashes.get("weights_sha256"))


def main(run_dir, mode, variant="smoke"):
    import metric.train as train
    ctx = context(run_dir, variant)
    if mode == "resume":
        return train.resume_training(ctx, ctx.run_dir / "checkpoints" / "last")
    if mode == "train-5":
        real, n = train.make_proxy_step, []

        def killable(*a):
            fn = real(*a)

            def step(*args):
                if len(n) == KILL_AFTER:
                    raise _Killed
                n.append(1)
                return fn(*args)
            return step
        train.make_proxy_step = killable
        try:
            train.run_training(ctx)
        except _Killed:
            return None
        raise AssertionError("run was not cut")
    return train.run_training(ctx)


if __name__ == "__main__":
    main(*sys.argv[1:])
