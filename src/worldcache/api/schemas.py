from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    scene_id: str
    question: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 42
    cache_level: str = "visual_embedding"


class GenerationOptions(BaseModel):
    max_new_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 42


class BranchItem(BaseModel):
    branch_id: str
    question: str = Field(min_length=1)


class BranchRequest(BaseModel):
    scene_id: str
    branches: list[BranchItem] = Field(min_length=1)
    generation_config: GenerationOptions = GenerationOptions()
    cache_level: str = "visual_embedding"
