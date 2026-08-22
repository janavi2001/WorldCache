from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .base import GenerationConfig, GenerationResult, StageMetrics, VLMAdapter


@dataclass(frozen=True)
class DeterministicVisualState:
    digest: bytes

    @property
    def nbytes(self) -> int:
        return len(self.digest)

    @property
    def shape(self) -> tuple[int]:
        return (len(self.digest),)


class DeterministicAdapter(VLMAdapter):
    """Fast, dependency-free adapter used for tests and smoke benchmarks.

    It is not a foundation model and benchmark reports label it accordingly.
    """

    def __init__(self, stage_delay_seconds: float = 0.0, work_factor: int = 4000) -> None:
        self.delay = stage_delay_seconds
        self.work_factor = work_factor
        self.preprocessing_calls = 0
        self.vision_calls = 0

    @staticmethod
    def _elapsed(start: int) -> float:
        return (time.perf_counter_ns() - start) / 1_000_000

    def preprocess_scene(self, scene: bytes) -> tuple[bytes, StageMetrics]:
        start = time.perf_counter_ns()
        self.preprocessing_calls += 1
        if self.delay:
            time.sleep(self.delay)
        processed = b"normalized:" + scene
        return processed, StageMetrics(preprocessing_ms=self._elapsed(start))

    def encode_scene(self, processed_scene: bytes) -> tuple[DeterministicVisualState, StageMetrics]:
        start = time.perf_counter_ns()
        self.vision_calls += 1
        if self.delay:
            time.sleep(self.delay * 4)
        digest = processed_scene
        for _ in range(self.work_factor):
            digest = hashlib.sha256(digest).digest()
        value = DeterministicVisualState(digest)
        return value, StageMetrics(
            vision_encoding_ms=self._elapsed(start), visual_tokens=1, vision_encoder_executions=1
        )

    def build_multimodal_prefix(
        self, visual_state: DeterministicVisualState, system_prompt: str
    ) -> bytes:
        return visual_state.digest + system_prompt.encode()

    def generate_from_scene(
        self,
        visual_state: DeterministicVisualState,
        question: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        start = time.perf_counter_ns()
        seed = visual_state.digest + question.encode() + str(generation_config.seed).encode()
        digest = hashlib.sha256(seed).hexdigest()
        answer = f"deterministic:{digest[:16]}"
        tokens = list(answer.encode())[: generation_config.max_new_tokens]
        elapsed = self._elapsed(start)
        metrics = StageMetrics(
            prefill_ms=elapsed * 0.35,
            time_to_first_token_ms=elapsed * 0.35,
            decode_ms=elapsed * 0.65,
            end_to_end_ms=elapsed,
            prompt_tokens=len(question.split()),
            visual_tokens=1,
            generated_tokens=len(tokens),
        )
        return GenerationResult(answer, tokens, metrics)

    def model_metadata(self) -> dict[str, object]:
        return {
            "model_id": "deterministic-test-adapter",
            "model_revision": "1",
            "processor_revision": "1",
            "dtype": "float32",
            "device": "cpu",
            "processor_config": {"normalization": "byte-prefix-v1"},
            "visual_tokenization": {"tokens": 1},
            "is_test_adapter": True,
            "supports_visual_embedding_cache": True,
        }

    def value_metadata(self, value: object) -> tuple[list[int], int, str]:
        if isinstance(value, DeterministicVisualState):
            return [len(value.digest)], value.nbytes, "cpu"
        return super().value_metadata(value)
