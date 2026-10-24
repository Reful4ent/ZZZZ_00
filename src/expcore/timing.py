"""Timing and memory measurements (spec §9)."""

import sys
import time
from typing import Any, Callable


def timed(fn: Callable, *args: Any) -> tuple[Any, float]:
    """(result, elapsed milliseconds) measured with perf_counter_ns."""
    t0 = time.perf_counter_ns()
    result = fn(*args)
    return result, (time.perf_counter_ns() - t0) / 1e6


def peak_memory_mb() -> tuple[float | None, str | None]:
    """(peak MB, None) via psutil when installed, else (None, reason_code)."""
    try:
        import psutil
    except ImportError:
        return None, "psutil_unavailable"
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)   # Windows
    if peak is None and sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if peak is None:
        return None, "peak_memory_unavailable"
    return peak / 2**20, None
