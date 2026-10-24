import numpy as np
import pytest


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
