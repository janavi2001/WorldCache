from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneWorkload:
    path: Path
    questions: list[str]


def load_workload(directory: str | Path) -> list[SceneWorkload]:
    root = Path(directory)
    manifest = root / "questions.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing workload manifest: {manifest}")
    data = json.loads(manifest.read_text())
    workloads = []
    for item in data["scenes"]:
        path = root / item["image"]
        if not path.is_file():
            raise FileNotFoundError(path)
        workloads.append(SceneWorkload(path, list(item["questions"])))
    if not workloads:
        raise ValueError("workload contains no scenes")
    return workloads
