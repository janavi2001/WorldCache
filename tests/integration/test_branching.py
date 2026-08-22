import unittest

from worldcache.cache.tiers import CacheLevel
from worldcache.models.deterministic import DeterministicAdapter
from worldcache.serving.engine import WorldCacheEngine


class BranchTests(unittest.TestCase):
    def test_visual_state_resolved_once(self):
        adapter = DeterministicAdapter(0)
        engine = WorldCacheEngine(adapter, 1024)
        sid = engine.register_scene(b"scene", CacheLevel.NONE, False)["scene_id"]
        result = engine.branch(sid, [("a", "left?"), ("b", "right?")])
        self.assertEqual(adapter.vision_calls, 1)
        self.assertEqual(len(result["branches"]), 2)
        engine.branch(sid, [("c", "forward?")])
        self.assertEqual(adapter.vision_calls, 1)


if __name__ == "__main__":
    unittest.main()
