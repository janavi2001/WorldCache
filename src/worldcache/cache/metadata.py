from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CacheMetadata:
    scene_id: str
    model_id: str
    model_revision: str
    processor_revision: str
    cache_level: str
    dtype: str
    shape: list[int]
    size_bytes: int
    device: str
    created_at: str = ""
    last_accessed_at: str = ""
    hit_count: int = 0
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        now = utc_now()
        self.created_at = self.created_at or now
        self.last_accessed_at = self.last_accessed_at or now
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

    def touch(self) -> None:
        self.last_accessed_at = utc_now()
        self.hit_count += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CacheMetadata:
        return cls(**value)
