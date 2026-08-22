from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

from worldcache.cache.tiers import CacheLevel
from worldcache.models.base import GenerationConfig
from worldcache.models.deterministic import DeterministicAdapter
from worldcache.serving.engine import WorldCacheEngine

from .report import write_svg_plot
from .runner import BenchmarkRunner
from .workload import load_workload


def _variant(base: dict[str, Any], root: Path, name: str, **changes: Any) -> dict[str, Any]:
    config = copy.deepcopy(base)
    for dotted, value in changes.items():
        section, key = dotted.split("__", 1)
        config[section][key] = value
    config["output"]["directory"] = str(root / name)
    return config


def run_suite(base: dict[str, Any]) -> dict[str, Any]:
    root = Path(base["output"]["directory"]) / "suite"
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {}

    experiments: dict[str, list[tuple[str, Any]]] = {
        "cache_levels": [
            ("none", "none"),
            ("preprocessing", "preprocessing"),
            ("visual_embedding", "visual_embedding"),
        ],
        "hit_rate": [(str(rate), rate) for rate in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "questions_per_scene": [(str(count), count) for count in (1, 2, 4, 8, 16)],
        # A test embedding is 32 bytes, so these retain 2, 5, or all scene entries.
        "memory_budget": [("small", 64), ("medium", 160), ("large", 64 * 1024**2)],
        "concurrency": [(str(count), count) for count in (1, 2, 4, 8)],
    }
    for experiment, values in experiments.items():
        rows = []
        for label, value in values:
            changes: dict[str, Any] = {}
            if experiment == "cache_levels":
                changes["cache__level"] = value
            elif experiment == "hit_rate":
                changes["workload__cache_hit_rate"] = value
            elif experiment == "questions_per_scene":
                changes["workload__questions_per_scene"] = value
            elif experiment == "memory_budget":
                changes["cache__max_bytes"] = value
            else:
                changes["workload__concurrency"] = value
            config = _variant(base, root, f"{experiment}_{label}", **changes)
            summary = BenchmarkRunner(config).run()
            summary["variant"] = value
            rows.append(summary)
        results[experiment] = rows

    workloads = load_workload(base["workload"]["dataset_path"])
    branch_rows = []
    for count in (1, 2, 4, 8, 16):
        adapter = DeterministicAdapter()
        engine = WorldCacheEngine(adapter, int(base["cache"]["max_bytes"]))
        scene = workloads[0]
        sid = engine.register_scene(scene.path.read_bytes(), CacheLevel.NONE, False)["scene_id"]
        questions = [
            (str(index), scene.questions[index % len(scene.questions)]) for index in range(count)
        ]
        started = time.perf_counter_ns()
        engine.branch(
            sid, questions, CacheLevel.VISUAL_EMBEDDING, GenerationConfig(**base["generation"])
        )
        cold_ms = (time.perf_counter_ns() - started) / 1_000_000
        started = time.perf_counter_ns()
        engine.branch(
            sid, questions, CacheLevel.VISUAL_EMBEDDING, GenerationConfig(**base["generation"])
        )
        hit_ms = (time.perf_counter_ns() - started) / 1_000_000
        branch_rows.append(
            {"variant": count, "cold_ms": cold_ms, "hit_ms": hit_ms, "speedup": cold_ms / hit_ms}
        )
    results["branches"] = branch_rows

    (root / "suite.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    hit = results["hit_rate"]
    write_svg_plot(
        root / "latency_vs_cache_hit_rate.svg",
        "Latency versus cache-hit rate",
        "Target hit rate",
        "Median latency (ms)",
        [(float(r["variant"]), float(r["median_latency_ms"])) for r in hit],
    )
    write_svg_plot(
        root / "throughput_vs_cache_hit_rate.svg",
        "Throughput versus cache-hit rate",
        "Target hit rate",
        "Requests/s",
        [(float(r["variant"]), float(r["requests_per_second"])) for r in hit],
    )
    write_svg_plot(
        root / "latency_vs_questions_per_scene.svg",
        "Latency versus questions per scene",
        "Questions per scene",
        "Median latency (ms)",
        [
            (float(r["variant"]), float(r["median_latency_ms"]))
            for r in results["questions_per_scene"]
        ],
    )
    write_svg_plot(
        root / "cache_memory_vs_hit_rate.svg",
        "Cache memory versus hit rate",
        "Target hit rate",
        "Peak cached bytes",
        [(float(r["variant"]), float(r["cache"]["peak_bytes"])) for r in hit],
    )
    write_svg_plot(
        root / "speedup_vs_branches.svg",
        "Speedup versus number of branches",
        "Branches",
        "Speedup",
        [(float(r["variant"]), float(r["speedup"])) for r in branch_rows],
    )
    stage_names = ["preprocessing_ms", "vision_encoding_ms", "prefill_ms", "decode_ms"]
    base_records = json.loads((root / "hit_rate_0.75" / "aggregate.json").read_text())
    # The aggregate has end-to-end values; stage means come from the raw JSONL.
    raw_rows = [
        json.loads(line)
        for line in (root / "hit_rate_0.75" / "requests.jsonl").read_text().splitlines()
    ]
    stage_points = [
        (float(index), float(statistics.mean(row[name] for row in raw_rows)))
        for index, name in enumerate(stage_names, 1)
    ]
    del base_records
    write_svg_plot(
        root / "latency_breakdown_by_stage.svg",
        "Latency breakdown: preprocessing, vision, prefill, decode",
        "Stage index",
        "Mean latency (ms)",
        stage_points,
    )
    return results
