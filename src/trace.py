"""Structured step logs.

A `Trace` collects every step of one operation (a correction request, an ingestion run) as a
list of JSON-friendly events. Each event is also forwarded to the standard `logging` module so it
shows up in the server console / server log buffer, and optionally to a `sink` callback that the
API uses to stream events to the frontend while the pipeline is still running.
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

logger = logging.getLogger("address_corrector")

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "success": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class Trace:
    def __init__(self, sink: Callable[[dict], None] | None = None, log: logging.Logger = logger):
        self.events: list[dict] = []
        self.timings: dict[str, float] = {}
        self._sink = sink
        self._log = log
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)

    def log(self, step: str, message: str, level: str = "info", data: dict[str, Any] | None = None) -> dict:
        event = {
            "seq": len(self.events) + 1,
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_ms": self.elapsed_ms,
            "level": level,
            "step": step,
            "message": message,
            "data": data,
        }
        self.events.append(event)
        self._log.log(LEVELS.get(level, logging.INFO), "[%s] %s", step, message)
        if self._sink is not None:
            self._sink(event)
        return event

    @contextmanager
    def timed(self, key: str):
        # Accumulates wall time under `key`; the yielded dict holds this block's duration afterwards
        timer = {"ms": 0.0}
        start = time.perf_counter()
        try:
            yield timer
        finally:
            timer["ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.timings[key] = round(self.timings.get(key, 0.0) + timer["ms"], 2)
