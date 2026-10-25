from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def smooth_image(seed: int, h: int = 48, w: int = 64) -> np.ndarray:
    """Непостоянное гладкое RGB-изображение в [0, 1] (имитация оригинала)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.empty((h, w, 3))
    for c in range(3):
        f = rng.uniform(1, 6, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        img[..., c] = 0.5 + 0.25 * np.sin(2 * np.pi * f[0] * xx + ph[0]) * np.cos(2 * np.pi * f[1] * yy + ph[1])
    img += rng.normal(0, 0.05, size=img.shape)
    return np.clip(img, 0.05, 0.95)


def _u8(img):
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def make_fake_kadid(root, n_originals: int, h: int = 48, w: int = 64, full_names: bool = False):
    """Фиктивная база в формате KADID-10k: images/Ixx.png, images/Ixx_yy_zz.png, dmos.csv.

    Искажения — простые фотометрические преобразования, только для проверки конвейера.
    """
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    lines = ["dist_img,ref_img,dmos,var"]
    for i in range(1, n_originals + 1):
        ref = smooth_image(i, h, w)
        Image.fromarray(_u8(ref)).save(img_dir / f"I{i:02d}.png")
        rng = np.random.default_rng(1000 + i)
        for t in range(1, 26):
            for lvl in range(1, 6):
                name = f"I{i:02d}_{t:02d}_{lvl:02d}.png"
                if full_names and t not in (7, 10, 16, 25):
                    # для проверки состава достаточно наличия файлов
                    (img_dir / name).write_bytes(b"")
                else:
                    k = lvl / 5
                    d = ref * (1 - 0.3 * k) + 0.15 * k * (t % 3) + rng.normal(0, 0.02 * k, ref.shape)
                    Image.fromarray(_u8(np.clip(d, 0, 1))).save(img_dir / name)
                lines.append(f"{name},I{i:02d}.png,{5 - lvl * 0.7:.3f},0.1")
    (root / "dmos.csv").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def img_a():
    return smooth_image(1)


@pytest.fixture
def img_b():
    return smooth_image(2)
