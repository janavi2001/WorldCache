import unittest

from worldcache.benchmark.correctness import compare_tensors


class CorrectnessMetricTests(unittest.TestCase):
    def test_error_metrics(self):
        result = compare_tensors([1.0, 2.0], [1.0, 2.001], atol=0.01, rtol=0)
        self.assertTrue(result.close)
        self.assertAlmostEqual(result.max_absolute_error, 0.001)
        self.assertAlmostEqual(result.mean_absolute_error, 0.0005)


if __name__ == "__main__":
    unittest.main()
