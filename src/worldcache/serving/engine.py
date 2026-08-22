from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from worldcache.cache.keys import CacheKeyConfig, cache_key, normalize_image_bytes, scene_id
from worldcache.cache.lru import ByteLRUCache
from worldcache.cache.metadata import CacheMetadata
from worldcache.cache.tiers import CacheLevel
from worldcache.models.base import GenerationConfig, StageMetrics, VLMAdapter
from worldcache.serving.instrumentation import MetricsCollector, RequestRecord
from worldcache.utils.profiling import profiling_range


@dataclass
class QueryResult:
    answer: str
    token_ids: list[int]
    cache_hit: bool
    cache_level: str
    metrics: StageMetrics
    request_id: str


@dataclass(frozen=True)
class RegisteredScene:
    encoded: bytes
    normalized: bytes


class SceneNotFoundError(KeyError):
    pass


class WorldCacheEngine:
    SYSTEM_PROMPT = "Answer questions about the physical scene accurately and concisely."

    def __init__(
        self,
        adapter: VLMAdapter,
        max_cache_bytes: int = 4 * 1024**3,
        ttl_seconds: float | None = None,
        max_scene_bytes: int = 512 * 1024**2,
    ) -> None:
        self.adapter = adapter
        self.cache = ByteLRUCache(max_cache_bytes, ttl_seconds)
        self.scenes = ByteLRUCache(max_scene_bytes)
        self.metrics = MetricsCollector()
        self._inference_lock = threading.RLock()

    def _key_config(self, level: CacheLevel) -> CacheKeyConfig:
        meta = self.adapter.model_metadata()
        return CacheKeyConfig(
            frame_sampling={"type": "single_image"},
            processor_config=dict(meta.get("processor_config", {})),
            model_id=str(meta["model_id"]),
            model_revision=str(meta.get("model_revision", "main")),
            processor_revision=str(meta.get("processor_revision", "main")),
            cache_level=level.value,
            dtype=str(meta.get("dtype", "unknown")),
            visual_tokenization=dict(meta.get("visual_tokenization", {})),
        )

    def _scene_id(self, normalized_scene: bytes) -> str:
        cfg = self._key_config(CacheLevel.PREPROCESSING)
        processing = {
            "frame_sampling": cfg.frame_sampling,
            "processor_config": cfg.processor_config,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "processor_revision": cfg.processor_revision,
            "dtype": cfg.dtype,
            "visual_tokenization": cfg.visual_tokenization,
        }
        return scene_id(normalized_scene, processing)

    def _cache_key(self, normalized_scene: bytes, level: CacheLevel) -> str:
        return f"{level.value}:{cache_key(normalized_scene, self._key_config(level))}"

    def _metadata(self, sid: str, level: CacheLevel, value: Any) -> CacheMetadata:
        model = self.adapter.model_metadata()
        shape, size, device = self.adapter.value_metadata(value)
        representation = {
            CacheLevel.PREPROCESSING: "pre_encoder_tensors",
            CacheLevel.VISUAL_EMBEDDING: "vision_encoder_output",
        }.get(level, level.value)
        return CacheMetadata(
            scene_id=sid,
            model_id=str(model["model_id"]),
            model_revision=str(model.get("model_revision", "main")),
            processor_revision=str(model.get("processor_revision", "main")),
            cache_level=level.value,
            dtype=str(model.get("dtype", "unknown")),
            shape=shape,
            size_bytes=size,
            device=device,
            extra={"representation": representation},
        )

    def register_scene(
        self, scene: bytes, cache_level: CacheLevel = CacheLevel.NONE, precompute: bool = True
    ) -> dict[str, Any]:
        if not scene:
            raise ValueError("scene is empty")
        normalized = normalize_image_bytes(scene)
        sid = self._scene_id(normalized)
        registered = RegisteredScene(bytes(scene), normalized)
        scene_meta = CacheMetadata(
            scene_id=sid,
            model_id="scene-registry",
            model_revision="1",
            processor_revision="1",
            cache_level="scene",
            dtype="bytes",
            shape=[len(scene)],
            size_bytes=len(scene) + len(normalized),
            device="cpu",
        )
        self.scenes.put(sid, registered, scene_meta)
        metrics = StageMetrics()
        hit = False
        cached_bytes = 0
        if precompute and cache_level != CacheLevel.NONE:
            _, hit, metrics, cached_bytes = self._resolve_visual(registered, sid, cache_level)
        return {
            "scene_id": sid,
            "cache_level": cache_level.value,
            "cache_hit": hit,
            "preprocessing_ms": metrics.preprocessing_ms,
            "vision_encoding_ms": metrics.vision_encoding_ms,
            "cached_bytes": cached_bytes,
        }

    def _resolve_visual(
        self, scene: RegisteredScene, sid: str, level: CacheLevel
    ) -> tuple[Any, bool, StageMetrics, int]:
        if level == CacheLevel.NONE:
            return scene.encoded, False, StageMetrics(), 0
        key = self._cache_key(scene.normalized, level)
        with profiling_range("cache_lookup"):
            cached = self.cache.get(key)
        if cached is not None:
            metadata = self.cache.metadata(key)
            with profiling_range("cache_copy"):
                value = cached  # GPU entries remain device-resident; no transfer is required.
            return value, True, StageMetrics(), metadata.size_bytes if metadata else 0
        with profiling_range("image_preprocessing"):
            processed, pre_metrics = self.adapter.preprocess_scene(scene.encoded)
        if level == CacheLevel.PREPROCESSING:
            metadata = self._metadata(sid, level, processed)
            self.cache.put(key, processed, metadata)
            return processed, False, pre_metrics, metadata.size_bytes
        if level == CacheLevel.PREFIX_KV:
            raise NotImplementedError("prefix-KV caching is not implemented in the MVP")
        with profiling_range("vision_encoder"):
            visual, encode_metrics = self.adapter.encode_scene(processed)
        pre_metrics.add(encode_metrics)
        metadata = self._metadata(sid, level, visual)
        self.cache.put(key, visual, metadata)
        return visual, False, pre_metrics, metadata.size_bytes

    def query(
        self,
        sid: str,
        question: str,
        cache_level: CacheLevel = CacheLevel.VISUAL_EMBEDDING,
        generation_config: GenerationConfig | None = None,
    ) -> QueryResult:
        if not question.strip():
            raise ValueError("question is empty")
        registered = self.scenes.get(sid)
        if registered is None:
            raise SceneNotFoundError(sid)
        config = generation_config or GenerationConfig()
        started = time.perf_counter_ns()
        with self._inference_lock:
            visual, hit, stage_metrics, _ = self._resolve_visual(registered, sid, cache_level)
            if cache_level == CacheLevel.NONE:
                generated = self.adapter.generate_uncached(registered.encoded, question, config)
            elif cache_level == CacheLevel.PREPROCESSING:
                generated = self.adapter.generate_from_processed(visual, question, config)
            else:
                generated = self.adapter.generate_from_scene(visual, question, config)
        stage_metrics.add(generated.metrics)
        stage_metrics.end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000
        request_id = str(uuid.uuid4())
        self.metrics.record(
            RequestRecord(request_id, sid, cache_level.value, hit, stage_metrics.to_dict())
        )
        return QueryResult(
            generated.answer, generated.token_ids, hit, cache_level.value, stage_metrics, request_id
        )

    def branch(
        self,
        sid: str,
        branches: list[tuple[str, str]],
        cache_level: CacheLevel = CacheLevel.VISUAL_EMBEDDING,
        generation_config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        registered = self.scenes.get(sid)
        if registered is None:
            raise SceneNotFoundError(sid)
        config = generation_config or GenerationConfig()
        with self._inference_lock:
            visual, hit, shared_metrics, _ = self._resolve_visual(registered, sid, cache_level)
            outputs = []
            for branch_id, question in branches:
                if cache_level == CacheLevel.PREPROCESSING:
                    result = self.adapter.generate_from_processed(visual, question, config)
                elif cache_level == CacheLevel.NONE:
                    result = self.adapter.generate_uncached(registered.encoded, question, config)
                else:
                    result = self.adapter.generate_from_scene(visual, question, config)
                outputs.append(
                    {
                        "branch_id": branch_id,
                        "answer": result.answer,
                        "metrics": result.metrics.to_dict(),
                    }
                )
        return {
            "scene_id": sid,
            "cache_level": cache_level.value,
            "cache_hit": hit,
            "shared_stage_metrics": shared_metrics.to_dict(),
            "branches": outputs,
        }

    def clear(self) -> None:
        self.cache.clear()
        self.scenes.clear()

    def cache_stats(self) -> dict[str, Any]:
        return self.cache.stats()

    def invalidate_representation(self, sid: str, level: CacheLevel) -> bool:
        """Remove one scene representation, primarily for controlled cold benchmarks."""
        registered = self.scenes.get(sid)
        if registered is None:
            raise SceneNotFoundError(sid)
        with self._inference_lock:
            return self.cache.delete(self._cache_key(registered.normalized, level))
