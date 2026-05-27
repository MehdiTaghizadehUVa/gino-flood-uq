"""Small timing helpers for serving diagnostics.

The timings emitted here are operational diagnostics, not scientific products.
They intentionally live outside the inference implementation so callers can
record end-to-end serving phases without coupling product code to wall-clock
measurement details.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class PerformanceTimer:
    """Accumulate wall-clock seconds for named serving phases."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._started_at = perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        self._seconds[str(name)] = self._seconds.get(str(name), 0.0) + max(0.0, float(seconds))

    def payload(self) -> dict[str, dict[str, float]]:
        phases = {name: {"seconds": float(seconds)} for name, seconds in sorted(self._seconds.items())}
        phases["total"] = {"seconds": max(0.0, perf_counter() - self._started_at)}
        return phases
