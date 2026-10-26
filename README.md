# WorldCache

WorldCache is an inference-serving and benchmarking system for repeated visual question answering.
It avoids recomputing image processing and vision features when multiple questions refer to the same
physical scene.

The project demonstrates a complete path from cache-key design and bounded storage to model
integration, REST serving, correctness checks, profiling, and reproducible benchmarks.

## Highlights

- FastAPI service for scene registration, querying, branching, health, cache statistics, and metrics.
- Thread-safe, byte-bounded LRU cache with TTL support, metadata, eviction accounting, and per-level
  statistics.
- Deterministic SHA-256 cache keys incorporating scene content and output-affecting model,
  processor, precision, and visual-tokenization configuration.
- Adapter boundary separating serving and caching logic from model-specific execution.
- Production adapter for `HuggingFaceTB/SmolVLM2-2.2B-Instruct`.
- Dependency-light deterministic adapter for CI, correctness checks, and benchmark-harness testing.
- Request-level latency, throughput, cache, GPU-memory, and vision-execution metrics.
- NVTX ranges and benchmark sweeps for cache level, hit rate, memory budget, concurrency, and
  branching.

The working cache levels are:

- `none`: complete uncached inference.
- `preprocessing`: reuse processor outputs and skip repeated image preprocessing.
- `visual_embedding`: reuse post-vision-encoder image states and skip the complete visual pipeline.

Prefix-KV reuse, video, multi-GPU serving, and future-world generation are explicit non-MVP items and
are not claimed as implemented.

## Architecture

`WorldCacheEngine` owns an in-memory scene registry and a byte-bounded representation cache. Cache
keys include normalized scene bytes plus model, processor, precision, and visual-tokenization
configuration, so incompatible revisions cannot reuse old representations.

The `VLMAdapter` interface isolates model-specific work into preprocessing, visual encoding, and
generation. The SmolVLM adapter uses a sentinel-based prompt split so the processor-generated image
token structure can be preserved while questions change. A cache hit can therefore go directly to
language generation with the image hidden states already resident on the device.

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

## Included benchmark results

The checked-in results are from the dependency-free deterministic adapter on CPU. They validate the
cache and benchmark machinery, but they are not representative of SmolVLM or GPU performance.
The main run used 200 requests, 8 questions per scene, a 75% target hit rate, and the
`visual_embedding` cache level. The observed hit rate was 72.5%:

| Metric | Result |
| --- | ---: |
| Median latency | 0.066 ms |
| P95 latency | 0.866 ms |
| Throughput | 3,489 requests/s |
| Hit vs. miss speedup | 13.23× |
| Vision compute avoided | 72.5% |
| Vision encoder executions avoided | 145 / 200 |
| Peak cached representation memory | 320 bytes |

In the cache-level sweep, median latency was 0.700 ms with no cache, 0.812 ms with preprocessing
caching, and 0.059 ms with visual-embedding caching. The complete raw requests, aggregate metrics,
and generated plots are available under [`results/experiment_001`](results/experiment_001), including
the [aggregate summary](results/experiment_001/aggregate.json) and [suite results](results/experiment_001/suite/suite.json).

For reportable VLM results, replace the deterministic model in `configs/benchmark.yaml` with the
SmolVLM model, run the suite on a controlled CUDA device, and report the resulting raw and aggregate
files alongside the hardware and model configuration.

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

## Current limitations

- The scene registry and representation cache are in-process and non-durable.
- Inference is serialized for the single-device adapter; true stream isolation and multi-GPU
  execution are not implemented.
- The current visual cache stores post-vision-encoder states, not language-model prefix KV state.
- The SmolVLM adapter reports vision and connector work together because that is the boundary exposed
  by the selected model API.
- The system provides qualitative VLM scene reasoning, not physically accurate future prediction.

See [docs/architecture.md](docs/architecture.md), [docs/methodology.md](docs/methodology.md), and
[docs/limitations.md](docs/limitations.md) for implementation and evaluation details.

## Project layout

```text
src/worldcache/
  api/          FastAPI application and request schemas
  cache/        cache levels, keys, metadata, and byte-bounded LRU
  models/       adapter interface, deterministic adapter, and SmolVLM adapter
  serving/      request orchestration, branching, and instrumentation
  benchmark/    workloads, runner, correctness, reports, and experiment suites
tests/          unit and integration coverage
data/           sample scenes and question manifest
configs/        server and benchmark configuration
docs/           architecture, methodology, and limitations
```

