"""Кеш представлений: память (LRU по байтам) + диск representation_cache/.

На диск сохраняются компактные представления (гистограммы, хеши) всех изображений
и все представления эталонов (оригиналов). Попиксельные представления запросов
по умолчанию не сохраняются: они читаются по одному разу, а на диске заняли бы
десятки гигабайт. Каталог метода включает хеш его конфигурации, запись
проверяется по размеру и mtime исходного файла.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from photometric.io import prepare_image
from photometric.methods import Representation, represent, resolve


class LazyImage:
    """Декодирует файл один раз, при первом обращении."""

    def __init__(self, path, color_profile="sRGB"):
        self.path = str(path)
        self.color_profile = color_profile
        self._img = None
        self._stat = None
        self.decode_ns = None

    def stat(self) -> os.stat_result:
        if self._stat is None:
            self._stat = os.stat(self.path)
        return self._stat

    def release(self) -> None:
        self._img = None

    def get(self) -> np.ndarray:
        if self._img is None:
            t0 = time.perf_counter_ns()
            self._img = prepare_image(self.path, self.color_profile)
            self.decode_ns = time.perf_counter_ns() - t0
        return self._img


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


class RepresentationCache:
    def __init__(self, root=None, memory_mb: float = 1024):
        self.root = Path(root) if root else None
        self.memory_limit = int(memory_mb * 1024 * 1024)
        self._mem: OrderedDict = OrderedDict()
        self._mem_bytes = 0
        self.stats = {"memory_hits": 0, "disk_hits": 0, "computed": 0}

    def _disk_path(self, method_id, chash, key) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{method_id}-{chash}" / f"{_safe(key)}.npz"

    def get(self, method_cfg, key: str, image: LazyImage, persist: bool) -> tuple[Representation, int]:
        """Возвращает (представление, время его построения в нс)."""
        spec, params, chash = resolve(method_cfg)
        mkey = (spec.method_id, chash, key)
        if mkey in self._mem:
            self._mem.move_to_end(mkey)
            self.stats["memory_hits"] += 1
            return self._mem[mkey]

        st = image.stat()
        disk = self._disk_path(spec.method_id, chash, key)
        if disk is not None and disk.is_file():
            try:
                with np.load(disk, allow_pickle=False) as z:
                    if int(z["src_size"]) == st.st_size and int(z["src_mtime_ns"]) == st.st_mtime_ns:
                        rep = Representation(spec.method_id, chash, z["data"], {"params": params})
                        item = (rep, int(z["repr_ns"]))
                        self.stats["disk_hits"] += 1
                        self._remember(mkey, item)
                        return item
            except (OSError, KeyError, ValueError):
                pass

        img = image.get()
        t0 = time.perf_counter_ns()
        rep = represent(img, method_cfg)
        repr_ns = time.perf_counter_ns() - t0
        self.stats["computed"] += 1
        if persist and disk is not None:
            disk.parent.mkdir(parents=True, exist_ok=True)
            tmp = disk.with_name(disk.stem + ".tmp.npz")
            np.savez(tmp, data=rep.data, repr_ns=repr_ns, src_size=st.st_size, src_mtime_ns=st.st_mtime_ns)
            os.replace(tmp, disk)
        item = (rep, repr_ns)
        self._remember(mkey, item)
        return item

    def _remember(self, mkey, item) -> None:
        size = item[0].nbytes
        if size > self.memory_limit:
            return
        self._mem[mkey] = item
        self._mem_bytes += size
        while self._mem_bytes > self.memory_limit:
            _, (old, _) = self._mem.popitem(last=False)
            self._mem_bytes -= old.nbytes

    def forget(self, key: str) -> None:
        """Удаляет из памяти все представления изображения (используется для запросов)."""
        for mkey in [k for k in self._mem if k[2] == key]:
            rep, _ = self._mem.pop(mkey)
            self._mem_bytes -= rep.nbytes
