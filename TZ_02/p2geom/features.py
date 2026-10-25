"""detect_describe (ГЕОМ-02, ГЕОМ-03).

Ключевые точки ищутся на яркостном представлении рабочего изображения
(длинная сторона <= max_side, только уменьшение с сохранением пропорций).
Коэффициенты A (исходные -> рабочие координаты) сохраняются; координаты точек
отдаются в обеих системах.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import FeatureConfig
from .homography import project, resize_crop_matrix


@dataclass
class Features:
    method: str
    status: str                       # OK | NOT_RUN
    kp_work: np.ndarray               # (N, 2) рабочие координаты
    kp_orig: np.ndarray               # (N, 2) исходные координаты
    desc: np.ndarray                  # (N, D)
    norm: str                         # l2 | hamming
    A: np.ndarray                     # исходные -> рабочие (включает resize и crop)
    orig_size: tuple[int, int]        # (W, H)
    work_size: tuple[int, int]
    seconds: float = 0.0
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.kp_work)


def to_luma_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] или uint8 -> яркость uint8 (ITU-R BT.601, как cv2)."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def prepare_work_image(rgb: np.ndarray, max_side: int,
                       crop: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Яркостное рабочее изображение и A (исходные -> рабочие)."""
    gray = to_luma_u8(rgb)
    H0, W0 = gray.shape
    if crop is not None:
        x0, y0, w, h = crop
        gray = gray[y0:y0 + h, x0:x0 + w]
    h, w = gray.shape
    s = min(1.0, max_side / float(max(w, h)))
    new = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    if new != (w, h):
        gray = cv2.resize(gray, new, interpolation=cv2.INTER_AREA)
    A = resize_crop_matrix((W0, H0), new, crop)
    return gray, A


def _cap_by_response(kps, desc, n: int):
    if len(kps) <= n:
        return kps, desc
    resp = np.array([k.response for k in kps])
    order = np.argsort(-resp, kind="stable")[:n]
    return [kps[i] for i in order], desc[order]


def detect_describe(rgb: np.ndarray, cfg: FeatureConfig,
                    crop: tuple[int, int, int, int] | None = None) -> Features:
    t0 = time.perf_counter()
    gray, A = prepare_work_image(rgb, cfg.max_side, crop)
    H0, W0 = rgb.shape[:2]
    info: dict = {}
    if cfg.method == "sift":
        det = cv2.SIFT_create(nfeatures=cfg.max_keypoints)
        kps, desc = det.detectAndCompute(gray, None)
        norm = "l2"
    elif cfg.method == "orb":
        det = cv2.ORB_create(nfeatures=cfg.max_keypoints, WTA_K=2)
        kps, desc = det.detectAndCompute(gray, None)
        norm = "hamming"
    elif cfg.method == "superpoint":
        from .superpoint import superpoint_detect
        pts, desc, info = superpoint_detect(gray, cfg)
        if info.get("status") != "OK":
            return Features("superpoint", "NOT_RUN", np.zeros((0, 2)), np.zeros((0, 2)),
                            np.zeros((0, 256), np.float32), "l2", A, (W0, H0),
                            (gray.shape[1], gray.shape[0]), time.perf_counter() - t0, info)
        kp_work = pts
        return _finish("superpoint", kp_work, desc, "l2", A, (W0, H0), gray, t0, info)
    else:
        raise ValueError(f"unknown feature method {cfg.method!r}")
    if desc is None or not kps:
        kps, desc = [], np.zeros((0, 128 if norm == "l2" else 32), np.float32 if norm == "l2" else np.uint8)
    kps, desc = _cap_by_response(list(kps), desc, cfg.max_keypoints)
    kp_work = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
    return _finish(cfg.method, kp_work, desc, norm, A, (W0, H0), gray, t0, info)


def _finish(method, kp_work, desc, norm, A, orig_size, gray, t0, info) -> Features:
    kp_orig, _ = project(np.linalg.inv(A), kp_work)
    return Features(method, "OK", kp_work, kp_orig, desc, norm, A, orig_size,
                    (gray.shape[1], gray.shape[0]), time.perf_counter() - t0, info)
