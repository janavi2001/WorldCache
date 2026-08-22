from __future__ import annotations

import os
from functools import lru_cache

from worldcache.models.base import VLMAdapter
from worldcache.models.deterministic import DeterministicAdapter
from worldcache.serving.engine import WorldCacheEngine


@lru_cache(maxsize=1)
def get_engine() -> WorldCacheEngine:
    adapter_name = os.getenv("WORLDCACHE_ADAPTER", "deterministic")
    max_bytes = int(os.getenv("WORLDCACHE_MAX_BYTES", str(4 * 1024**3)))
    adapter: VLMAdapter
    if adapter_name == "smolvlm":
        from worldcache.models.selected_vlm import SmolVLMAdapter

        adapter = SmolVLMAdapter(
            model_id=os.getenv("WORLDCACHE_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
            revision=os.getenv("WORLDCACHE_MODEL_REVISION", "main"),
            processor_revision=os.getenv("WORLDCACHE_PROCESSOR_REVISION", "main"),
            dtype=os.getenv("WORLDCACHE_DTYPE", "float16"),
            device=os.getenv("WORLDCACHE_DEVICE", "cuda:0"),
        )
    else:
        adapter = DeterministicAdapter()
    return WorldCacheEngine(adapter, max_cache_bytes=max_bytes)
