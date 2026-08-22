from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def timed(synchronize: Callable[[], None] | None = None) -> Iterator[Callable[[], float]]:
    if synchronize:
        synchronize()
    started = time.perf_counter_ns()
    value = [0.0]
    try:
        yield lambda: value[0]
    finally:
        if synchronize:
            synchronize()
        value[0] = (time.perf_counter_ns() - started) / 1_000_000
