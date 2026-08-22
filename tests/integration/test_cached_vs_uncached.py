import unittest

from worldcache.cache.tiers import CacheLevel
from worldcache.models.base import GenerationConfig
from worldcache.models.deterministic import DeterministicAdapter
from worldcache.serving.engine import WorldCacheEngine


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.adapter = DeterministicAdapter(stage_delay_seconds=0.001)
        self.engine = WorldCacheEngine(self.adapter, max_cache_bytes=1024)
        self.scene = b"self-created-scene"
        self.sid = self.engine.register_scene(self.scene, CacheLevel.NONE, False)["scene_id"]
        self.config = GenerationConfig(max_new_tokens=64, temperature=0, seed=42)

    def test_cached_tokens_equal_uncached(self):
        uncached = self.engine.query(self.sid, "Where is red?", CacheLevel.NONE, self.config)
        first = self.engine.query(
            self.sid, "Where is red?", CacheLevel.VISUAL_EMBEDDING, self.config
        )
        second = self.engine.query(
            self.sid, "Where is red?", CacheLevel.VISUAL_EMBEDDING, self.config
        )
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(uncached.token_ids, second.token_ids)
        self.assertEqual(uncached.answer, second.answer)

    def test_preprocessing_and_embedding_equivalence(self):
        one, _ = self.adapter.preprocess_scene(self.scene)
        two, _ = self.adapter.preprocess_scene(self.scene)
        self.assertEqual(one, two)
        first, _ = self.adapter.encode_scene(one)
        second, _ = self.adapter.encode_scene(two)
        self.assertEqual(first.digest, second.digest)

    def test_hit_skips_vision_and_is_faster(self):
        self.engine.query(self.sid, "q1", CacheLevel.VISUAL_EMBEDDING, self.config)
        calls = self.adapter.vision_calls
        hit = self.engine.query(self.sid, "q2", CacheLevel.VISUAL_EMBEDDING, self.config)
        self.assertEqual(self.adapter.vision_calls, calls)
        cold = self.engine.query(self.sid, "q2", CacheLevel.NONE, self.config)
        self.assertLess(hit.metrics.end_to_end_ms, cold.metrics.end_to_end_ms)

    def test_revision_changes_scene_id(self):
        original = self.sid
        self.adapter.model_metadata = lambda: {
            "model_id": "deterministic-test-adapter",
            "model_revision": "2",
            "processor_revision": "1",
            "dtype": "float32",
            "device": "cpu",
            "processor_config": {},
            "visual_tokenization": {},
        }
        changed = self.engine.register_scene(self.scene, CacheLevel.NONE, False)["scene_id"]
        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
