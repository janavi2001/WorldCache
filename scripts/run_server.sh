#!/usr/bin/env bash
set -euo pipefail
export WORLDCACHE_ADAPTER="${WORLDCACHE_ADAPTER:-smolvlm}"
exec uvicorn worldcache.api.app:app --host 0.0.0.0 --port 8000

