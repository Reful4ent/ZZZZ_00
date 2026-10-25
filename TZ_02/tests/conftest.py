import cv2
import numpy as np
import pytest
from skimage import data as skdata


@pytest.fixture(scope="session")
def texture():
    """Цветное изображение 320×240 с богатой текстурой."""
    img = skdata.astronaut().astype(np.float64) / 255.0
    return cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)


def random_h(rng, size, amp=0.05):
    W, H = size
    src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    dst = src + rng.uniform(-amp, amp, (4, 2)).astype(np.float32) * np.array([W, H], np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)
