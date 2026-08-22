#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/nsight
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output=results/nsight/worldcache \
  --force-overwrite=true \
  python -m worldcache.benchmark --config configs/benchmark.yaml

