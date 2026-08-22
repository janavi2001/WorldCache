#!/usr/bin/env bash
set -euo pipefail
python -m worldcache.benchmark --config configs/benchmark.yaml --suite
