# WorldCache

> WorldCache is an inference and benchmarking system for reusing visual representations across repeated physical-scene reasoning requests. It caches preprocessing results, vision embeddings, and optionally multimodal prefix state so that vision-language models can answer multiple questions about the same image or video without repeating unchanged computation. The project evaluates latency, throughput, GPU memory, cache efficiency, and deterministic equivalence between cached and uncached execution.

WorldCache provides an instrumented FastAPI server, a byte-bounded thread-safe LRU, a model-adapter
boundary, deterministic correctness checks, a repeated-scene benchmark suite, Prometheus metrics,
and Nsight ranges. The production adapter targets
`HuggingFaceTB/SmolVLM2-2.2B-Instruct`; the small dependency-free adapter is for CI and benchmark
harness validation only.

## What is implemented

- Level 0: complete uncached inference.
- Level 1: cached processor tensors; image decoding and normalization are skipped on a hit.
- Level 2: cached image hidden states; preprocessing, pixel transfer, and the model's image-feature
  calculation are skipped on a hit.
- Exact deterministic SHA-256 keys include all output-affecting model and processor configuration.
- One-GPU REST serving, multiple questions per registered image, and branch evaluation with a single
  shared visual lookup.
- Request-level stage metrics, cache metrics, JSONL/JSON reports, experiment sweeps, SVG graphs, and
  NVTX ranges.

Prefix-KV reuse, video, multi-GPU serving, and future-world generation are explicit non-MVP items and
are not claimed as implemented.

## Setup

Python 3.10+ and a CUDA-capable PyTorch environment are recommended for the real model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

For cache/unit/API development without model weights:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Start the server

The provided launcher selects the SmolVLM adapter. Model weights are downloaded by Transformers on
the first run.

```bash
WORLDCACHE_DEVICE=cuda:0 WORLDCACHE_DTYPE=float16 ./scripts/run_server.sh
```

For a CPU-only functional smoke server:

```bash
WORLDCACHE_ADAPTER=deterministic uvicorn worldcache.api.app:app --port 8000
python examples/client.py data/sample_scenes/scene_01.ppm
```

The deterministic adapter does not produce meaningful visual answers; it exists to verify routing,
caching, and exact output reuse without a model download.

## API

Register and precompute a visual embedding:

```bash
curl -s http://localhost:8000/v1/scenes \
  -F image=@data/sample_scenes/scene_01.ppm \
  -F cache_level=visual_embedding \
  -F precompute=true
```

Then use the returned `scene_id`:

```bash
curl -s http://localhost:8000/v1/query \
  -H 'content-type: application/json' \
  -d '{"scene_id":"sha256:...","question":"Is the path blocked?","temperature":0.0,"cache_level":"visual_embedding"}'
```

Other endpoints are `POST /v1/branch`, `GET /v1/cache/stats`, `DELETE /v1/cache`, `GET /health`,
and `GET /metrics`. To observe first-miss/second-hit behavior, register with `precompute=false`.

## Benchmarks and graphs

The single configured run writes `requests.jsonl` and `aggregate.json`:

```bash
python -m worldcache.benchmark --config configs/benchmark.yaml
```

All required MVP sweeps and six SVG plots are produced with:

```bash
python -m worldcache.benchmark --config configs/benchmark.yaml --suite
```

The suite covers cache level, target hit rate (0/25/50/75/100%), questions per scene
(1/2/4/8/16), memory budget, concurrency (1/2/4/8), and branches (1/2/4/8/16). Change the model
block to the SmolVLM ID and `cuda:0` before collecting reportable VLM numbers. Keep deterministic
generation (`temperature: 0.0`) for equivalence checks.

Profile the configured benchmark with:

```bash
./scripts/profile_nsight.sh
```

Inspect `image_preprocessing`, `host_to_device_transfer`, `vision_encoder`, `cache_lookup`,
`cache_copy`, and `language_prefill_and_token_decoding` ranges. A Level-2 hit should contain no
`vision_encoder` range on the critical path.

## Reproducible verification

```bash
ruff check .
ruff format --check .
pytest -q
```

Tests require cached and uncached token IDs to match, verify representation equivalence and key
invalidation, and assert that a warmed Level-2 hit makes zero additional vision-encoder calls.
Normal CI does not enforce a strict latency ratio because shared hosts are noisy.

## Reporting results

Do not substitute the included deterministic smoke benchmark for GPU/VLM evidence. After completing
the CUDA suite, report the measured result in this form only:

> Across repeated physical-scene reasoning workloads, WorldCache reduced median time to first token by X%, increased throughput by Y×, and avoided Z% of repeated vision-encoder executions at a cache-hit rate of H%. Cached execution produced the same deterministic token outputs as uncached execution. Performance gains depended on the number of questions per scene, cache-memory budget, and representation being cached.

Fill in X/Y/Z/H from the generated raw and aggregate GPU results. See
[architecture](docs/architecture.md), [methodology](docs/methodology.md), and
[limitations](docs/limitations.md) before interpreting results.

