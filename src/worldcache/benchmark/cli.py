from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import BenchmarkRunner


def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the benchmark CLI") from exc
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark repeated-scene VLM inference")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--suite", action="store_true", help="run all required MVP experiment sweeps"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.suite:
        from .suite import run_suite

        summary = run_suite(config)
    else:
        summary = BenchmarkRunner(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
