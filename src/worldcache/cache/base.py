from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .metadata import CacheMetadata


class CacheManager(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def put(self, key: str, value: Any, metadata: CacheMetadata) -> bool: ...

    @abstractmethod
    def contains(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...

    @abstractmethod
    def set_memory_budget(self, number_of_bytes: int) -> None: ...
