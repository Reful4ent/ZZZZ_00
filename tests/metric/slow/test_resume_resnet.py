"""ПРИЁМ-М3 on ResNet50 224 px with ImageNet weights (16 fixture classes); run with `-m slow`."""
import socket
from pathlib import Path

import pytest

from tests.metric.smoke.test_resume import assert_same_runs, make_runs

pytestmark = [pytest.mark.slow, pytest.mark.tf,
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _weights_available():
    from metric.model import _WEIGHTS_FILE
    if (Path.home() / ".keras" / "models" / _WEIGHTS_FILE).is_file():
        return True
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


def test_10_equals_5_restore_5_resnet(tmp_path_factory):
    pytest.importorskip("tensorflow")
    if not _weights_available():
        pytest.skip("ImageNet ResNet50 weights not cached and no network")
    runs = make_runs(tmp_path_factory, "resnet")
    assert_same_runs(runs)
    from tests.metric.smoke.test_resume import _fingerprint
    assert _fingerprint(runs["b"] / "training_steps/part-001.parquet") == runs["part1"]
