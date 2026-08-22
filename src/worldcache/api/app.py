from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from worldcache.cache.tiers import CacheLevel
from worldcache.models.base import GenerationConfig
from worldcache.serving.engine import SceneNotFoundError, WorldCacheEngine

from .dependencies import get_engine
from .schemas import BranchRequest, QueryRequest


def create_app(engine: WorldCacheEngine | None = None) -> FastAPI:
    application = FastAPI(title="WorldCache", version="0.1.0")

    def dependency() -> WorldCacheEngine:
        return engine or get_engine()

    @application.post("/v1/scenes")
    async def register_scene(
        image: UploadFile = File(...),
        label: str | None = Form(default=None),
        cache_level: str = Form(default="visual_embedding"),
        precompute: bool = Form(default=True),
        service: WorldCacheEngine = Depends(dependency),
    ) -> dict:
        del label  # labels are intentionally not part of output-affecting cache keys
        try:
            level = CacheLevel.parse(cache_level)
            return service.register_scene(await image.read(), level, precompute)
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.post("/v1/query")
    def query(request: QueryRequest, service: WorldCacheEngine = Depends(dependency)) -> dict:
        try:
            result = service.query(
                request.scene_id,
                request.question,
                CacheLevel.parse(request.cache_level),
                GenerationConfig(request.max_new_tokens, request.temperature, request.seed),
            )
        except SceneNotFoundError as exc:
            raise HTTPException(status_code=404, detail="scene not found or evicted") from exc
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "request_id": result.request_id,
            "answer": result.answer,
            "cache_hit": result.cache_hit,
            "cache_level": result.cache_level,
            **result.metrics.to_dict(),
        }

    @application.post("/v1/branch")
    def branch(request: BranchRequest, service: WorldCacheEngine = Depends(dependency)) -> dict:
        try:
            config = request.generation_config
            return service.branch(
                request.scene_id,
                [(item.branch_id, item.question) for item in request.branches],
                CacheLevel.parse(request.cache_level),
                GenerationConfig(config.max_new_tokens, config.temperature, config.seed),
            )
        except SceneNotFoundError as exc:
            raise HTTPException(status_code=404, detail="scene not found or evicted") from exc
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.get("/v1/cache/stats")
    def cache_stats(service: WorldCacheEngine = Depends(dependency)) -> dict:
        return service.cache_stats()

    @application.delete("/v1/cache")
    def clear_cache(service: WorldCacheEngine = Depends(dependency)) -> dict:
        service.clear()
        return {"cleared": True}

    @application.get("/health")
    def health(service: WorldCacheEngine = Depends(dependency)) -> dict:
        metadata = service.adapter.model_metadata()
        return {"status": "ok", "model": metadata, "gpu": metadata.get("device", "cpu")}

    @application.get("/metrics", response_class=PlainTextResponse)
    def metrics(service: WorldCacheEngine = Depends(dependency)) -> str:
        return service.metrics.prometheus(service.cache_stats())

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("worldcache.api.app:app", host="0.0.0.0", port=8000)
