from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def aggregate(records: list[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    latencies = [float(row["end_to_end_ms"]) for row in records]
    ttfts = [float(row.get("time_to_first_token_ms", 0.0)) for row in records]
    generated = sum(int(row["generated_tokens"]) for row in records)
    avoided = sum(
        1 for row in records if row["cache_hit"] and row["cache_level"] == "visual_embedding"
    )
    baseline_executions = len(records)
    cached_executions = sum(int(row["vision_encoder_executions"]) for row in records)
    cached_mb = max((float(row.get("cached_bytes", 0)) for row in records), default=0.0) / 1024**2
    miss_latency = [float(r["end_to_end_ms"]) for r in records if not r["cache_hit"]]
    hit_latency = [float(r["end_to_end_ms"]) for r in records if r["cache_hit"]]
    saved_ms = (
        (statistics.mean(miss_latency) - statistics.mean(hit_latency))
        if miss_latency and hit_latency
        else 0.0
    )
    miss_vision = [
        float(r["vision_encoding_ms"]) for r in records if int(r["vision_encoder_executions"])
    ]
    return {
        "requests": len(records),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "mean_time_to_first_token_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "median_time_to_first_token_ms": statistics.median(ttfts) if ttfts else 0.0,
        "requests_per_second": len(records) / duration_seconds if duration_seconds else 0.0,
        "output_tokens_per_second": generated / duration_seconds if duration_seconds else 0.0,
        "cache_hit_rate": sum(bool(r["cache_hit"]) for r in records) / len(records)
        if records
        else 0.0,
        "vision_encoder_executions": cached_executions,
        "vision_encoder_executions_avoided": avoided,
        "vision_compute_avoided": 1 - cached_executions / baseline_executions
        if baseline_executions
        else 0.0,
        "gpu_hours_saved_estimate": (statistics.mean(miss_vision) * avoided / 3_600_000)
        if miss_vision
        else 0.0,
        "cache_efficiency_ms_per_mb": saved_ms / cached_mb if cached_mb else 0.0,
        "speedup_hit_vs_miss": statistics.mean(miss_latency) / statistics.mean(hit_latency)
        if hit_latency and miss_latency
        else 1.0,
        "max_gpu_memory_allocated": max(
            (int(r.get("gpu_memory_allocated", 0)) for r in records), default=0
        ),
        "max_gpu_memory_reserved": max(
            (int(r.get("gpu_memory_reserved", 0)) for r in records), default=0
        ),
        "stage_mean_ms": {
            name: statistics.mean(float(row.get(name, 0.0)) for row in records) if records else 0.0
            for name in (
                "preprocessing_ms",
                "host_to_device_ms",
                "vision_encoding_ms",
                "projection_ms",
                "prefill_ms",
                "decode_ms",
            )
        },
    }


def write_results(directory: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    raw = "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    (directory / "requests.jsonl").write_text(raw)
    (directory / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_svg_plot(
    path: Path, title: str, x_label: str, y_label: str, points: list[tuple[float, float]]
) -> None:
    width, height, pad = 720, 420, 60
    if not points:
        points = [(0.0, 0.0)]
    xs, ys = zip(*points, strict=True)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmax = xmax if xmax != xmin else xmin + 1
    ymax = ymax if ymax != ymin else ymin + 1
    coords = []
    for x, y in points:
        px = pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        py = height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
        coords.append(f"{px:.1f},{py:.1f}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{title}</text>\n'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
        f'y2="{height - pad}" stroke="black"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>\n'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="#2563eb" '
        'stroke-width="3"/>\n'
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" '
        f'font-family="sans-serif">{x_label}</text>'
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" '
        f'text-anchor="middle" font-family="sans-serif">{y_label}</text>\n'
        "</svg>"
    )
    path.write_text(svg)
