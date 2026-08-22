from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int = 42


@dataclass
class StageMetrics:
    preprocessing_ms: float = 0.0
    host_to_device_ms: float = 0.0
    vision_encoding_ms: float = 0.0
    projection_ms: float = 0.0
    prefill_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    decode_ms: float = 0.0
    end_to_end_ms: float = 0.0
    prompt_tokens: int = 0
    visual_tokens: int = 0
    generated_tokens: int = 0
    gpu_memory_allocated: int = 0
    gpu_memory_reserved: int = 0
    vision_encoder_executions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def add(self, other: StageMetrics) -> None:
        for name in (
            "preprocessing_ms",
            "host_to_device_ms",
            "vision_encoding_ms",
            "projection_ms",
            "prefill_ms",
            "decode_ms",
            "end_to_end_ms",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.time_to_first_token_ms = other.time_to_first_token_ms or self.time_to_first_token_ms
        for name in (
            "prompt_tokens",
            "visual_tokens",
            "generated_tokens",
            "vision_encoder_executions",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.gpu_memory_allocated = max(self.gpu_memory_allocated, other.gpu_memory_allocated)
        self.gpu_memory_reserved = max(self.gpu_memory_reserved, other.gpu_memory_reserved)


@dataclass
class GenerationResult:
    answer: str
    token_ids: list[int]
    metrics: StageMetrics


class VLMAdapter(ABC):
    @abstractmethod
    def preprocess_scene(self, scene: bytes) -> tuple[Any, StageMetrics]: ...

    @abstractmethod
    def encode_scene(self, processed_scene: Any) -> tuple[Any, StageMetrics]: ...

    @abstractmethod
    def build_multimodal_prefix(self, visual_state: Any, system_prompt: str) -> Any: ...

    @abstractmethod
    def generate_from_scene(
        self, visual_state: Any, question: str, generation_config: GenerationConfig
    ) -> GenerationResult: ...

    def generate_from_processed(
        self, processed_scene: Any, question: str, generation_config: GenerationConfig
    ) -> GenerationResult:
        visual_state, encode_metrics = self.encode_scene(processed_scene)
        result = self.generate_from_scene(visual_state, question, generation_config)
        result.metrics.add(encode_metrics)
        return result

    def generate_uncached(
        self, scene: bytes, question: str, generation_config: GenerationConfig
    ) -> GenerationResult:
        processed, preprocessing = self.preprocess_scene(scene)
        result = self.generate_from_processed(processed, question, generation_config)
        result.metrics.add(preprocessing)
        return result

    @abstractmethod
    def model_metadata(self) -> dict[str, Any]: ...

    def value_metadata(self, value: Any) -> tuple[list[int], int, str]:
        size = getattr(value, "nbytes", None)
        if size is None and hasattr(value, "numel") and hasattr(value, "element_size"):
            size = value.numel() * value.element_size()
        if size is None and isinstance(value, bytes):
            size = len(value)
        return (
            list(getattr(value, "shape", [])),
            int(size or 0),
            str(getattr(value, "device", "cpu")),
        )
