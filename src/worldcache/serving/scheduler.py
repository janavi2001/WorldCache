from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class RequestScheduler:
    """Serializes access to adapters that use a single non-thread-safe GPU stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def run(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return operation(*args, **kwargs)
