"""Tests for reusable continuous-attribute metrics."""

import math
import unittest

from legonet.eval.regression_metrics import compute_regression_metrics


class RegressionMetricsTests(unittest.TestCase):
    """Characterize formulas extracted from the legacy evaluator."""

    def test_calculates_regression_summary(self) -> None:
        metrics = compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 5.0])

        self.assertAlmostEqual(metrics.mean_absolute_error, 1.0)
        self.assertAlmostEqual(metrics.mean_squared_error, 5.0 / 3.0)
        self.assertAlmostEqual(metrics.mean_relative_error, 7.0 / 18.0)
        self.assertAlmostEqual(metrics.one_minus_fvu, -1.5)
        self.assertEqual(metrics.evaluated_samples, 3)
        self.assertEqual(metrics.coverage, 1.0)

    def test_uses_caller_relative_error_policy(self) -> None:
        metrics = compute_regression_metrics(
            [0.0, 2.0],
            [4.0, 1.0],
            relative_errors=[0.5],
        )

        self.assertEqual(metrics.mean_relative_error, 0.5)
        self.assertEqual(metrics.evaluated_samples, 2)

    def test_empty_input_has_explicitly_unavailable_metrics(self) -> None:
        metrics = compute_regression_metrics([], [])

        self.assertEqual(metrics.evaluated_samples, 0)
        self.assertIsNone(metrics.mean_absolute_error)
        self.assertIsNone(metrics.mean_relative_error)
        self.assertIsNone(metrics.coverage)

    def test_constant_targets_preserve_undefined_one_minus_fvu(self) -> None:
        metrics = compute_regression_metrics([2.0, 2.0], [1.0, 3.0])

        self.assertTrue(math.isnan(metrics.one_minus_fvu))

    def test_roots_compatibility_preserves_zero_variance_division(self) -> None:
        metrics = compute_regression_metrics(
            [2.0, 2.0],
            [1.0, 3.0],
            preserve_zero_variance_division=True,
        )

        self.assertEqual(metrics.one_minus_fvu, float("-inf"))

    def test_rejects_misaligned_observations(self) -> None:
        with self.assertRaises(ValueError):
            compute_regression_metrics([1.0, 2.0], [1.0])

    def test_rejects_excess_relative_errors(self) -> None:
        with self.assertRaises(ValueError):
            compute_regression_metrics([1.0], [1.0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
