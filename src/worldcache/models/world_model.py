from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WorldModelAdapter(ABC):
    """Extension point for a future generative world-model backend; not an MVP implementation."""

    @abstractmethod
    def encode_history(self, video_or_images: Any) -> Any: ...

    @abstractmethod
    def generate_future(
        self, encoded_history: Any, action_condition: Any, generation_config: Any
    ) -> Any: ...
