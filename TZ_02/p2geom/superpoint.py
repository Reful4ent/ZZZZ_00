"""Адаптер SuperPoint (ГЕОМ-03): официальные предобученные веса MagicLeap
(superpoint_v1.pth), дескрипторы L2, далее та же геометрическая проверка.

Сравнительный компонент, не код авторской статьи. PyTorch можно держать в
отдельном окружении (requirements-superpoint.txt). Если torch или веса
недоступны, либо SHA-256 не совпадает с заданным, статус NOT_RUN — замена
другим детектором под этим именем не выполняется.
Архитектура и постобработка повторяют demo_superpoint.py исходного репозитория
(NMS реализован через max-pooling).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import FeatureConfig


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_info(cfg: FeatureConfig) -> dict:
    return {"weights_path": cfg.superpoint_weights, "url": cfg.superpoint_weights_url,
            "sha256_expected": cfg.superpoint_weights_sha256, "license": cfg.superpoint_weights_license,
            "commit": cfg.superpoint_weights_commit}


_MODEL_CACHE: dict = {}


def _build_net(torch):
    nn = torch.nn

    class SuperPointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
            self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
            self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
            self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
            self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
            self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
            self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
            self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
            self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
            self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
            self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
            self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
            self.convDb = nn.Conv2d(c5, d1, 1, 1, 0)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1b(self.relu(self.conv1a(x)))))
            x = self.pool(self.relu(self.conv2b(self.relu(self.conv2a(x)))))
            x = self.pool(self.relu(self.conv3b(self.relu(self.conv3a(x)))))
            x = self.relu(self.conv4b(self.relu(self.conv4a(x))))
            semi = self.convPb(self.relu(self.convPa(x)))
            desc = self.convDb(self.relu(self.convDa(x)))
            return semi, desc / torch.norm(desc, p=2, dim=1, keepdim=True)

    return SuperPointNet()


def superpoint_detect(gray: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """gray uint8 рабочего изображения -> (точки N×2 в рабочих координатах, дескрипторы N×256, info)."""
    info = weights_info(cfg)
    try:
        import torch
    except ImportError:
        return np.zeros((0, 2)), np.zeros((0, 256), np.float32), {**info, "status": "NOT_RUN",
                                                                   "reason": "torch not installed"}
    path = Path(cfg.superpoint_weights) if cfg.superpoint_weights else None
    if path is None or not path.exists():
        return np.zeros((0, 2)), np.zeros((0, 256), np.float32), {**info, "status": "NOT_RUN",
                                                                   "reason": "weights file not found"}
    key = str(path)
    if key not in _MODEL_CACHE:
        sha = file_sha256(path)
        if cfg.superpoint_weights_sha256 and sha != cfg.superpoint_weights_sha256.lower():
            return np.zeros((0, 2)), np.zeros((0, 256), np.float32), {**info, "status": "NOT_RUN",
                                                                       "reason": "sha256 mismatch", "sha256": sha}
        net = _build_net(torch)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        _MODEL_CACHE[key] = (net, sha)
    net, sha = _MODEL_CACHE[key]
    info.update(sha256=sha)
    # Кадрирование до кратного 8 (правый/нижний край) — входит в A как crop без сдвига.
    H, W = gray.shape
    Hc, Wc = (H // 8) * 8, (W // 8) * 8
    img = gray[:Hc, :Wc].astype(np.float32) / 255.0
    with torch.no_grad():
        inp = torch.from_numpy(img)[None, None]
        semi, coarse = net(inp)
        dense = torch.softmax(semi[0], dim=0)[:-1]                       # без dustbin
        heat = torch.nn.functional.pixel_shuffle(dense[None], 8)[0, 0]   # (Hc, Wc)
        r = cfg.superpoint_nms_dist
        pooled = torch.nn.functional.max_pool2d(heat[None, None], 2 * r + 1, 1, r)[0, 0]
        keep = (heat == pooled) & (heat >= cfg.superpoint_conf_thresh)
        b = cfg.superpoint_border
        keep[:b, :] = keep[-b:, :] = False
        keep[:, :b] = keep[:, -b:] = False
        ys, xs = torch.nonzero(keep, as_tuple=True)
        scores = heat[ys, xs]
        order = torch.argsort(scores, descending=True, stable=True)[:cfg.max_keypoints]
        ys, xs = ys[order], xs[order]
        pts = torch.stack([xs, ys], 1).float()
        grid = pts.clone()
        grid[:, 0] = grid[:, 0] / (Wc / 2.0) - 1.0
        grid[:, 1] = grid[:, 1] / (Hc / 2.0) - 1.0
        desc = torch.nn.functional.grid_sample(coarse, grid[None, None], align_corners=True)[0, :, 0].T
        desc = desc / torch.norm(desc, p=2, dim=1, keepdim=True)
    info.update(status="OK", n=int(len(pts)))
    return pts.numpy().astype(np.float64), desc.numpy().astype(np.float32), info
