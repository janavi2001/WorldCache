# Benchmark methodology

Run a warm-up request before measurement. Use the same model weights, revision, prompt template,
dtype, seed, temperature, and output limit for cached and uncached paths. CUDA stage boundaries call
`torch.cuda.synchronize`, and the production adapter streams output so time to first generated text
is measured directly. Raw request rows are written to `requests.jsonl`; aggregate percentiles,
throughput, token rate, cache efficiency, and vision executions avoided go to `aggregate.json`.

The checked-in default benchmark uses the dependency-free deterministic adapter as a smoke and CI
benchmark. Its results validate the harness and cache mechanics, but are not evidence of VLM or GPU
performance. Publication-quality results must set the model ID and CUDA device, run all required
sweeps on a controlled GPU, and retain the Nsight reports alongside the JSONL data.

Correctness compares processed values exactly in the test adapter, embedding digests exactly, and
generated token IDs exactly. For SmolVLM, add hardware integration comparisons with `atol=1e-3` and
`rtol=1e-3` for float16 hidden states while still requiring identical greedy token IDs.

