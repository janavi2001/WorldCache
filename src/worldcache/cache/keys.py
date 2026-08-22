from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def normalize_image_bytes(encoded: bytes) -> bytes:
    """Return canonical RGB pixels when Pillow is present, otherwise preserve opaque bytes."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(encoded)).convert("RGB")
        header = canonical_json({"mode": "RGB", "width": image.width, "height": image.height})
        return header + b"\0" + image.tobytes()
    except (ImportError, OSError, ValueError):
        return encoded


@dataclass(frozen=True)
class CacheKeyConfig:
    frame_sampling: dict[str, Any]
    processor_config: dict[str, Any]
    model_id: str
    model_revision: str
    processor_revision: str
    cache_level: str
    dtype: str
    visual_tokenization: dict[str, Any]


def scene_id(normalized_scene_bytes: bytes, processing: dict[str, Any]) -> str:
    digest = hashlib.sha256(normalized_scene_bytes + b"\0" + canonical_json(processing)).hexdigest()
    return f"sha256:{digest}"


def cache_key(normalized_scene_bytes: bytes, config: CacheKeyConfig) -> str:
    digest = hashlib.sha256(
        normalized_scene_bytes + b"\0" + canonical_json(asdict(config))
    ).hexdigest()
    return f"sha256:{digest}"


def prefix_cache_key(base_key: str, token_ids: list[int], positions: list[int], mask: Any) -> str:
    payload = {"base_key": base_key, "token_ids": token_ids, "positions": positions, "mask": mask}
    return f"sha256:{hashlib.sha256(canonical_json(payload)).hexdigest()}"
