import unittest

from worldcache.benchmark.report import aggregate, percentile


class MetricTests(unittest.TestCase):
    def test_percentiles_and_aggregate(self):
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        records = [
            {
                "end_to_end_ms": 10,
                "generated_tokens": 2,
                "cache_hit": False,
                "cache_level": "visual_embedding",
                "vision_encoder_executions": 1,
                "vision_encoding_ms": 5,
                "cached_bytes": 1024**2,
            },
            {
                "end_to_end_ms": 5,
                "generated_tokens": 2,
                "cache_hit": True,
                "cache_level": "visual_embedding",
                "vision_encoder_executions": 0,
                "vision_encoding_ms": 0,
                "cached_bytes": 1024**2,
            },
        ]
        result = aggregate(records, 1.0)
        self.assertEqual(result["requests_per_second"], 2)
        self.assertEqual(result["vision_compute_avoided"], 0.5)
        self.assertEqual(result["speedup_hit_vs_miss"], 2)


if __name__ == "__main__":
    unittest.main()
