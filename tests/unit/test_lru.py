import time
import unittest

from worldcache.cache.lru import ByteLRUCache
from worldcache.cache.metadata import CacheMetadata


def metadata(scene, size, level="visual_embedding"):
    return CacheMetadata(scene, "m", "r", "p", level, "float32", [size], size, "cpu")


class LRUTests(unittest.TestCase):
    def test_lru_order_and_byte_eviction(self):
        cache = ByteLRUCache(6)
        cache.put("visual_embedding:a", b"aaa", metadata("a", 3))
        cache.put("visual_embedding:b", b"bbb", metadata("b", 3))
        self.assertEqual(cache.get("visual_embedding:a"), b"aaa")
        cache.put("visual_embedding:c", b"ccc", metadata("c", 3))
        self.assertFalse(cache.contains("visual_embedding:b"))
        self.assertTrue(cache.contains("visual_embedding:a"))
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_oversized_entry_is_rejected(self):
        cache = ByteLRUCache(2)
        self.assertFalse(cache.put("x", b"abc", metadata("x", 3)))
        self.assertEqual(cache.stats()["entries"], 0)

    def test_resize_budget(self):
        cache = ByteLRUCache(10)
        cache.put("a", b"aaaa", metadata("a", 4))
        cache.put("b", b"bbbb", metadata("b", 4))
        cache.set_memory_budget(4)
        self.assertEqual(cache.stats()["current_bytes"], 4)
        self.assertFalse(cache.contains("a"))

    def test_ttl(self):
        cache = ByteLRUCache(10, ttl_seconds=0.001)
        cache.put("visual_embedding:a", b"a", metadata("a", 1))
        time.sleep(0.003)
        self.assertIsNone(cache.get("visual_embedding:a"))

    def test_clear_releases_references(self):
        cache = ByteLRUCache(10)
        cache.put("a", b"a", metadata("a", 1))
        cache.clear()
        self.assertEqual(cache.stats()["current_bytes"], 0)
        self.assertEqual(cache.stats()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
