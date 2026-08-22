from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

from .base import CacheManager
from .metadata import CacheMetadata


@dataclass
class _Entry:
    value: Any
    metadata: CacheMetadata
    expires_at: float | None


class ByteLRUCache(CacheManager):
    """Thread-safe byte-bounded LRU cache suitable for CPU or GPU tensors."""

    def __init__(self, max_bytes: int, ttl_seconds: float | None = None) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._freed_bytes = 0
        self._levels: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "evictions": 0, "entries": 0, "bytes": 0}
        )

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                if ":" in key:
                    self._levels[key.split(":", 1)[0]]["misses"] += 1
                return None
            if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
                level = entry.metadata.cache_level
                self._remove(key, eviction=True)
                self._misses += 1
                self._levels[level]["misses"] += 1
                return None
            self._entries.move_to_end(key)
            entry.metadata.touch()
            self._hits += 1
            self._levels[entry.metadata.cache_level]["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, metadata: CacheMetadata) -> bool:
        if metadata.size_bytes < 0:
            raise ValueError("entry size cannot be negative")
        with self._lock:
            if key in self._entries:
                self._remove(key, eviction=False)
            if metadata.size_bytes > self._max_bytes:
                return False
            while self._entries and self._bytes + metadata.size_bytes > self._max_bytes:
                self._remove(next(iter(self._entries)), eviction=True)
            expiry = time.monotonic() + self._ttl if self._ttl is not None else None
            self._entries[key] = _Entry(value, metadata, expiry)
            self._bytes += metadata.size_bytes
            level = self._levels[metadata.cache_level]
            level["entries"] += 1
            level["bytes"] += metadata.size_bytes
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            return True

    def contains(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
                self._remove(key, eviction=True)
                return False
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key, eviction=False)

    def _remove(self, key: str, eviction: bool) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        size = entry.metadata.size_bytes
        level = self._levels[entry.metadata.cache_level]
        self._bytes -= size
        level["entries"] -= 1
        level["bytes"] -= size
        if eviction:
            self._evictions += 1
            self._freed_bytes += size
            level["evictions"] += 1
        entry.value = None
        del entry
        return True

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._remove(key, eviction=False)
            gc.collect()  # release Python references; never force torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            requests = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / requests if requests else 0.0,
                "evictions": self._evictions,
                "entries": len(self._entries),
                "current_bytes": self._bytes,
                "peak_bytes": self._peak_bytes,
                "freed_bytes": self._freed_bytes,
                "max_bytes": self._max_bytes,
                "by_level": {k: dict(v) for k, v in self._levels.items()},
            }

    def set_memory_budget(self, number_of_bytes: int) -> None:
        if number_of_bytes < 0:
            raise ValueError("memory budget must be non-negative")
        with self._lock:
            self._max_bytes = number_of_bytes
            while self._entries and self._bytes > number_of_bytes:
                self._remove(next(iter(self._entries)), eviction=True)

    def metadata(self, key: str) -> CacheMetadata | None:
        with self._lock:
            entry = self._entries.get(key)
            return entry.metadata if entry else None
