"""In-memory ring buffer of server log records, exposed to the frontend via /api/logs."""

import logging
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._records: deque[dict] = deque(maxlen=capacity)
        self._next_id = 1
        self._records_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry: dict[str, Any] = {
                "time": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
                "level": record.levelname.lower(),
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exception"] = logging.Formatter().formatException(record.exc_info)
            with self._records_lock:
                entry["id"] = self._next_id
                self._next_id += 1
                self._records.append(entry)
        except Exception:
            self.handleError(record)

    def since(self, after_id: int = 0, limit: int = 500) -> list[dict]:
        with self._records_lock:
            records = [entry for entry in self._records if entry["id"] > after_id]
        return records[-limit:]

    @property
    def last_id(self) -> int:
        with self._records_lock:
            return self._next_id - 1


memory_handler = MemoryLogHandler()


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # Bangla on Windows consoles
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console)
    for name in ("", "uvicorn"):  # uvicorn's loggers don't propagate to the root logger
        target = logging.getLogger(name)
        if memory_handler not in target.handlers:
            target.addHandler(memory_handler)
