import importlib.util
import unittest


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("fastapi", "pydantic", "httpx", "multipart")),
    "API test dependencies are not installed",
)
class APITests(unittest.TestCase):
    def test_register_query_stats_and_clear(self):
        from fastapi.testclient import TestClient

        from worldcache.api.app import create_app
        from worldcache.models.deterministic import DeterministicAdapter
        from worldcache.serving.engine import WorldCacheEngine

        engine = WorldCacheEngine(DeterministicAdapter(0), 1024)
        client = TestClient(create_app(engine))
        registered = client.post(
            "/v1/scenes",
            files={"image": ("scene.ppm", b"scene", "application/octet-stream")},
            data={"cache_level": "visual_embedding", "precompute": "false"},
        )
        self.assertEqual(registered.status_code, 200)
        sid = registered.json()["scene_id"]
        payload = {
            "scene_id": sid,
            "question": "What is visible?",
            "cache_level": "visual_embedding",
        }
        first = client.post("/v1/query", json=payload).json()
        second = client.post("/v1/query", json=payload).json()
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["answer"], second["answer"])
        self.assertIn("worldcache_cache_hits_total", client.get("/metrics").text)
        self.assertEqual(client.delete("/v1/cache").json(), {"cleared": True})
