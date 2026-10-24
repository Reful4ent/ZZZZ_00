"""python -m metric train|embed|search --config … — skeleton; TensorFlow is imported only inside handlers."""

import argparse
from typing import Sequence

from expcore.cli_common import command_parser, execute_run, run_cli
import metric.params  # noqa: F401  (registers the params schema)

MODULE = "metric"
COMMANDS = ("train", "embed", "search")


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


def _handle(args: argparse.Namespace) -> None:
    deterministic = _enable_tf_determinism()   # before init_seeds, so TF gets seeded too
    execute_run(MODULE, args.command, args, query=True, deterministic_ops=deterministic)


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(_handle, argv, parser=command_parser(MODULE, COMMANDS))
