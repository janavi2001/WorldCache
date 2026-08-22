from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from worldcache.cache.tiers import CacheLevel
from worldcache.models.base import GenerationConfig, VLMAdapter
from worldcache.models.deterministic import DeterministicAdapter
from worldcache.serving.engine import WorldCacheEngine
from worldcache.utils.gpu import GPUUtilizationSampler

from .report import aggregate, write_results
from .workload import load_workload


class BenchmarkRunner:
    def __init__(self, config: dict[str, Any], engine: WorldCacheEngine | None = None) -> None:
        self.config = config
        model_cfg = config["model"]
        if engine is None:
            adapter: VLMAdapter
            if model_cfg["id"] != "deterministic-test-adapter":
                from worldcache.models.selected_vlm import SmolVLMAdapter

                adapter = SmolVLMAdapter(
                    model_cfg["id"],
                    model_cfg.get("revision", "main"),
                    model_cfg.get("processor_revision", "main"),
                    model_cfg["dtype"],
                    model_cfg["device"],
                )
            else:
                adapter = DeterministicAdapter()
            engine = WorldCacheEngine(adapter, int(config["cache"]["max_bytes"]))
        self.engine = engine

    def run(self) -> dict[str, Any]:
        workload_cfg = self.config["workload"]
        workloads = load_workload(workload_cfg["dataset_path"])
        level = CacheLevel.parse(self.config["cache"]["level"])
        generation = GenerationConfig(**self.config["generation"])
        total = int(workload_cfg["total_requests"])
        concurrency = int(workload_cfg.get("concurrency", 1))
        target_hit_rate = float(workload_cfg.get("cache_hit_rate", 1.0))
        questions_per_scene = int(workload_cfg.get("questions_per_scene", 8))
        rng = random.Random(generation.seed)
        registered = [
            (
                self.engine.register_scene(item.path.read_bytes(), CacheLevel.NONE, False)[
                    "scene_id"
                ],
                item,
            )
            for item in workloads
        ]

        # Warm up model kernels without retaining reusable representations.
        warm_sid, warm_item = registered[0]
        self.engine.query(warm_sid, warm_item.questions[0], CacheLevel.NONE, generation)
        self.engine.cache.clear()

        jobs = []
        for index in range(total):
            sid, item = registered[(index // max(1, questions_per_scene)) % len(registered)]
            question = item.questions[index % min(questions_per_scene, len(item.questions))]
            force_miss = rng.random() >= target_hit_rate
            jobs.append((sid, question, force_miss))

        def execute(job: tuple[str, str, bool]) -> dict[str, Any]:
            sid, question, force_miss = job
            if force_miss:
                # Remove only this representation so unrelated warm scenes remain reusable.
                self.engine.invalidate_representation(sid, level)
            result = self.engine.query(sid, question, level, generation)
            return {
                "request_id": result.request_id,
                "scene_id": sid,
                "cache_level": level.value,
                "cache_hit": result.cache_hit,
                "cached_bytes": self.engine.cache.stats()["current_bytes"],
                **result.metrics.to_dict(),
            }

        gpu_sampler = GPUUtilizationSampler()
        gpu_sampler.start()
        started = time.perf_counter()
        if concurrency == 1:
            records = [execute(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                records = list(pool.map(execute, jobs))
        duration = time.perf_counter() - started
        gpu_utilization = gpu_sampler.stop()
        summary = aggregate(records, duration)
        summary["gpu_utilization"] = gpu_utilization
        cache_stats = self.engine.cache_stats()
        summary["cache"] = cache_stats
        summary["eviction_rate"] = (
            cache_stats["evictions"] / (cache_stats["evictions"] + cache_stats["entries"])
            if cache_stats["evictions"] + cache_stats["entries"]
            else 0.0
        )
        summary["memory_per_cache_entry"] = (
            cache_stats["current_bytes"] / cache_stats["entries"] if cache_stats["entries"] else 0.0
        )
        summary["config"] = self.config
        output = Path(self.config["output"]["directory"])
        write_results(output, records, summary)
        return summary
