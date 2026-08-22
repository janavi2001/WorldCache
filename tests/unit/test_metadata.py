import unittest

from worldcache.cache.metadata import CacheMetadata


class MetadataTests(unittest.TestCase):
    def test_round_trip_and_touch(self):
        item = CacheMetadata("s", "m", "r", "p", "visual_embedding", "f16", [1, 2], 4, "cpu")
        restored = CacheMetadata.from_dict(item.to_dict())
        restored.touch()
        self.assertEqual(restored.hit_count, 1)
        self.assertEqual(restored.shape, [1, 2])

    def test_reject_negative_size(self):
        with self.assertRaises(ValueError):
            CacheMetadata("s", "m", "r", "p", "x", "x", [], -1, "cpu")


if __name__ == "__main__":
    unittest.main()
