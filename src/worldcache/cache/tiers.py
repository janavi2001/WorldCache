from enum import Enum


class CacheLevel(str, Enum):
    NONE = "none"
    PREPROCESSING = "preprocessing"
    VISUAL_EMBEDDING = "visual_embedding"
    PREFIX_KV = "prefix_kv"

    @classmethod
    def parse(cls, value: str) -> "CacheLevel":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unsupported cache level: {value}") from exc
