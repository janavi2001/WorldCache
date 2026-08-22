import unittest

from worldcache.cache.lru import ByteLRUCache
from worldcache.cache.metadata import CacheMetadata


class EvictionTests(unittest.TestCase):
    def test_level_stats(self):
        cache = ByteLRUCache(4)
        meta = CacheMetadata("s", "m", "r", "p", "preprocessing", "u8", [4], 4, "cpu")
        cache.put("preprocessing:a", b"1234", meta)
        cache.get("preprocessing:a")
        cache.get("visual_embedding:missing")
        stats = cache.stats()
        self.assertEqual(stats["by_level"]["preprocessing"]["hits"], 1)
        self.assertEqual(stats["by_level"]["visual_embedding"]["misses"], 1)


if __name__ == "__main__":
    unittest.main()
