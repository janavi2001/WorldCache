import unittest

from worldcache.cache.keys import CacheKeyConfig, cache_key, prefix_cache_key, scene_id


def config(**changes):
    values = dict(
        frame_sampling={"type": "single_image"},
        processor_config={"size": 384},
        model_id="model",
        model_revision="abc",
        processor_revision="def",
        cache_level="visual_embedding",
        dtype="float16",
        visual_tokenization={"patch": 14},
    )
    values.update(changes)
    return CacheKeyConfig(**values)


class CacheKeyTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(cache_key(b"image", config()), cache_key(b"image", config()))

    def test_configuration_invalidates(self):
        base = cache_key(b"image", config())
        self.assertNotEqual(base, cache_key(b"image", config(model_revision="new")))
        self.assertNotEqual(base, cache_key(b"image", config(processor_config={"size": 512})))
        self.assertNotEqual(base, cache_key(b"image", config(dtype="bfloat16")))

    def test_images_do_not_collide(self):
        self.assertNotEqual(cache_key(b"image-a", config()), cache_key(b"image-b", config()))

    def test_scene_and_prefix_keys(self):
        self.assertTrue(scene_id(b"x", {"size": 1}).startswith("sha256:"))
        self.assertNotEqual(
            prefix_cache_key("base", [1, 2], [0, 1], [1, 1]),
            prefix_cache_key("base", [1, 3], [0, 1], [1, 1]),
        )


if __name__ == "__main__":
    unittest.main()
