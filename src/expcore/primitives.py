"""Library primitives are named and checked, not trusted (spec §10, ОБЩ-03).

PRIMITIVES lists {name, library, check_id}; primitive_versions() adds the installed version and goes
into environment.json / run.json. run_check(check_id) verifies a primitive's behaviour on synthetic
data and raises AssertionError on a mismatch, PrimitiveUnavailable when its library is missing.
"""

import importlib.metadata
import importlib.util
from typing import Callable

import numpy as np

PRIMITIVES: tuple[dict, ...] = (
    {"name": "SIFT", "library": "opencv-python", "check_id": "PRIM-SIFT"},
    {"name": "ORB", "library": "opencv-python", "check_id": "PRIM-SIFT"},
    {"name": "cv2.cvtColor RGB<->Lab", "library": "opencv-python", "check_id": "PRIM-COLOR"},
    {"name": "skimage.color.rgb2lab/lab2rgb", "library": "scikit-image", "check_id": "PRIM-COLOR"},
    {"name": "numpy.linalg.lstsq/svd", "library": "numpy", "check_id": "PRIM-LINALG"},
    {"name": "scipy.linalg.lstsq", "library": "scipy", "check_id": "PRIM-LINALG"},
    {"name": "cv2.findHomography", "library": "opencv-python", "check_id": "PRIM-LINALG"},
    {"name": "keras.optimizers.Adam", "library": "tensorflow", "check_id": "PRIM-ADAM"},
    {"name": "keras.applications.ResNet50", "library": "tensorflow", "check_id": "PRIM-BACKBONE"},
)

# Tolerances of the control checks.
SIFT_MIN_KEYPOINTS = 50          # textured 256×256 board yields several hundred; 50 flags a broken detector
SIFT_MIN_MATCHES = 20            # ratio-test matches needed to estimate H robustly
SIFT_H_TOL_PX = 1.0              # rotation by np.rot90 is exact, so RANSAC H must map corners within a pixel
COLOR_ROUNDTRIP_TOL = 5e-3       # OpenCV float32 Lab approximates the sRGB gamma with splines: 2.45e-3 measured
                                 # on OpenCV 5.0 (~0.6 of 255 levels); skimage (float64) is checked at 1e-6
COLOR_AGREEMENT_TOL = 0.5        # ΔE76 between OpenCV and scikit-image on reference colours
LINALG_TOL = 1e-9
HOMOGRAPHY_EXACT_TOL = 1e-6


class PrimitiveUnavailable(RuntimeError):
    """The primitive's library is not installed; the check is skipped with this reason."""


_EXTRA: list[dict] = []
_EXTRA_CHECKS: dict[str, Callable[[], None]] = {}


def register_primitive(name: str, library: str, check_id: str, check: Callable[[], None]) -> None:
    """Programs (TZ 01–03) add their own primitives and control checks here."""
    if check_id in _CHECKS or (check_id in _EXTRA_CHECKS and _EXTRA_CHECKS[check_id] is not check):
        raise ValueError(f"check {check_id!r} already registered")
    _EXTRA.append({"name": name, "library": library, "check_id": check_id})
    _EXTRA_CHECKS[check_id] = check


def _version(library: str) -> str | None:
    try:
        return importlib.metadata.version(library)
    except importlib.metadata.PackageNotFoundError:
        return None


def primitive_versions() -> list[dict]:
    return [{**p, "version": _version(p["library"])} for p in (*PRIMITIVES, *_EXTRA)]


def _textured_board(size: int = 256, cells: int = 8) -> np.ndarray:
    """Checkerboard whose cells have distinct grey levels (fixed seed), so a 90° rotation is unambiguous."""
    rng = np.random.Generator(np.random.PCG64(2026))
    levels = rng.integers(30, 226, (cells, cells))
    board = np.kron(levels, np.ones((size // cells, size // cells))).astype(np.uint8)
    board[(np.add.outer(np.arange(size), np.arange(size)) // (size // cells)) % 2 == 0] //= 3
    return board


def _check_sift() -> None:
    import cv2
    img = _textured_board()
    rot = np.ascontiguousarray(np.rot90(img))   # counter-clockwise: (x, y) → (y, W−1−x)
    w = img.shape[1]
    h_true = np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=float)
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(img, None)
    kp2, d2 = sift.detectAndCompute(rot, None)
    assert min(len(kp1), len(kp2)) >= SIFT_MIN_KEYPOINTS, f"SIFT found {len(kp1)}/{len(kp2)} keypoints"
    assert len(cv2.ORB_create().detect(img, None)) >= SIFT_MIN_KEYPOINTS, "ORB found too few keypoints"
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    assert len(good) >= SIFT_MIN_MATCHES, f"only {len(good)} ratio-test matches"
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    h_est, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    assert h_est is not None, "findHomography failed"
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, w - 1], [0, w - 1]]).reshape(-1, 1, 2)
    err = np.abs(cv2.perspectiveTransform(corners, h_est) - cv2.perspectiveTransform(corners, h_true)).max()
    assert err < SIFT_H_TOL_PX, f"estimated homography off by {err:.3f} px"


def _check_color() -> None:
    import cv2
    from skimage.color import lab2rgb, rgb2lab
    white = np.ones((1, 1, 3), np.float32)
    assert abs(cv2.cvtColor(white, cv2.COLOR_RGB2Lab)[0, 0, 0] - 100) < 1e-3, "OpenCV: D65 white is not L=100"
    assert abs(rgb2lab(white.astype(float))[0, 0, 0] - 100) < 1e-6, "scikit-image: D65 white is not L=100"
    rng = np.random.Generator(np.random.PCG64(2026))
    colours = rng.random((1, 64, 3)).astype(np.float32)
    back = cv2.cvtColor(cv2.cvtColor(colours, cv2.COLOR_RGB2Lab), cv2.COLOR_Lab2RGB)
    assert np.abs(back - colours).max() < COLOR_ROUNDTRIP_TOL, "OpenCV Lab round trip drifts"
    assert np.abs(lab2rgb(rgb2lab(colours.astype(float))) - colours).max() < 1e-6, "skimage round trip drifts"
    delta = np.linalg.norm(cv2.cvtColor(colours, cv2.COLOR_RGB2Lab) - rgb2lab(colours.astype(float)), axis=-1)
    assert delta.max() < COLOR_AGREEMENT_TOL, f"OpenCV and scikit-image Lab differ by ΔE {delta.max():.3f}"


def _check_linalg() -> None:
    import cv2
    import scipy.linalg
    a = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4], [1, 0, 1], [3, 2, 1], [1, 1, 1]], dtype=float)
    x_true = np.array([1.5, -2.0, 0.25])
    b = a @ x_true
    assert np.abs(np.linalg.lstsq(a, b, rcond=None)[0] - x_true).max() < LINALG_TOL, "numpy lstsq"
    assert np.abs(scipy.linalg.lstsq(a, b)[0] - x_true).max() < LINALG_TOL, "scipy lstsq"
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    assert np.abs(u @ np.diag(s) @ vt - a).max() < LINALG_TOL, "SVD reconstruction"
    h_true = np.array([[1.1, 0.05, 3.0], [-0.02, 0.95, -2.0], [1e-4, 2e-4, 1.0]])
    src = np.float64([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40], [20, 70]])
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_true).reshape(-1, 2)
    h_est, _ = cv2.findHomography(src, dst, 0)
    assert np.abs(h_est / h_est[2, 2] - h_true).max() < HOMOGRAPHY_EXACT_TOL, "findHomography on exact points"


def _tf():
    if importlib.util.find_spec("tensorflow") is None:
        raise PrimitiveUnavailable("TensorFlow is not installed (extra `metric`)")
    import tensorflow as tf
    return tf


def _check_adam() -> None:
    tf = _tf()
    lr, eps, theta0 = 0.1, 1e-7, 1.0
    theta = tf.Variable(theta0, dtype=tf.float64)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
    with tf.GradientTape() as tape:
        loss = theta * theta
    grad = tape.gradient(loss, theta)   # ĝ = 2θ₀
    opt.apply_gradients([(grad, theta)])
    g = 2 * theta0
    expected = theta0 - lr * g / (abs(g) + eps)   # first step: m̂ = ĝ, v̂ = ĝ²
    assert abs(float(theta.numpy()) - expected) < 1e-6, f"Adam step {float(theta.numpy())} != {expected}"


def _check_backbone() -> None:
    tf = _tf()
    model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3),
                                           pooling="avg")
    assert tuple(model.output_shape) == (None, 2048), f"ResNet50 output {model.output_shape}"
    assert model.count_params() == 23_587_712, f"ResNet50 has {model.count_params()} parameters"


_CHECKS: dict[str, Callable[[], None]] = {
    "PRIM-SIFT": _check_sift, "PRIM-COLOR": _check_color, "PRIM-LINALG": _check_linalg,
    "PRIM-ADAM": _check_adam, "PRIM-BACKBONE": _check_backbone,
}


def run_check(check_id: str) -> None:
    check = _CHECKS.get(check_id) or _EXTRA_CHECKS.get(check_id)
    if check is None:
        raise KeyError(f"unknown primitive check {check_id!r}")
    check()
