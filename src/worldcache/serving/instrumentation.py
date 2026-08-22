from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    scene_id: str
    cache_level: str
    cache_hit: bool
    metrics: dict[str, Any]


class MetricsCollector:
    def __init__(self, max_records: int = 10_000) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, item: RequestRecord) -> None:
        with self._lock:
            self._records.append(item)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(record) for record in self._records]

    def prometheus(self, cache_stats: dict[str, Any]) -> str:
        lines = [
            "# HELP worldcache_cache_hits_total Cache hits.",
            "# TYPE worldcache_cache_hits_total counter",
            f"worldcache_cache_hits_total {cache_stats['hits']}",
            "# HELP worldcache_cache_misses_total Cache misses.",
            "# TYPE worldcache_cache_misses_total counter",
            f"worldcache_cache_misses_total {cache_stats['misses']}",
            "# HELP worldcache_cache_bytes Cached bytes.",
            "# TYPE worldcache_cache_bytes gauge",
            f"worldcache_cache_bytes {cache_stats['current_bytes']}",
            "# HELP worldcache_cache_evictions_total Cache evictions.",
            "# TYPE worldcache_cache_evictions_total counter",
            f"worldcache_cache_evictions_total {cache_stats['evictions']}",
        ]
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps(self.records(), sort_keys=True)
