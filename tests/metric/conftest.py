import numpy as np
import pytest

import metric  # noqa: F401  (imported once here, so the pop below sticks for the session)
from expcore import registry


def pytest_collection_finish(session):
    # Collecting tests/metric imports `metric`, which registers the p3 builder process-wide (spec 3.4 item 3);
    # TZ00 tests build manifests with no program builders, so the session starts without it. Metric tests that
    # need it install it explicitly (tests.metric.world, real_pairs=True); registration itself is checked in a
    # fresh interpreter (test_cli.test_pair_builder_registered).
    registry._PAIR_BUILDERS.pop("p3", None)


@pytest.fixture
def tf_or_skip():
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")
    return tf


@pytest.fixture
def rng():
    return np.random.default_rng(0)
