"""Per-stage wall time and peak memory.

Runtime has to be reported in the demo and compute efficiency is 10% of the
score, so knowing the per-stage breakdown is how you find what to cut when you
are over budget.
"""

from __future__ import annotations

import logging
import resource
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# ru_maxrss is bytes on macOS and kilobytes on Linux.
import sys as _sys
_RSS_SCALE = 1.0 / (1024.0 * 1024.0) if _sys.platform == "darwin" else 1.0 / 1024.0


def peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE


class Timer:
    """Accumulates named stage timings for the output meta block."""

    def __init__(self) -> None:
        self.stages: List[Tuple[str, float]] = []
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages.append((name, elapsed))
            log.info("stage %-22s %6.2f s  (peak RSS %.0f MB)",
                     name, elapsed, peak_memory_mb())

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self._start

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, seconds in self.stages:
            out[name] = round(out.get(name, 0.0) + seconds, 3)
        return out
