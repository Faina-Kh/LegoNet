"""Tests for matched-object per-image attribute summaries."""

import unittest

from legonet.eval.per_image_attribute_metrics import (
    compute_roots_per_image_metrics,
)


class RootsPerImageMetricsTests(unittest.TestCase):
    """Verify that false-positive records do not enter attribute metrics."""

    def test_computes_metrics_from_matched_records_only(self) -> None:
        records = {
            "TRL_gt": [2.0, 4.0, -1.0],
            "TRL_pred": [1.0, 5.0, 100.0],
            "dia_gt": [1.0, 3.0, -1.0],
            "dia_pred": [1.0, 2.0, 100.0],
            "color_gt": [0, 1, -1],
            "color_pred": [0, 0, 1],
        }

        metrics = compute_roots_per_image_metrics(records)

        self.assertEqual(metrics.trl.evaluated_samples, 2)
        self.assertAlmostEqual(metrics.trl.one_minus_fvu, 0.0)
        self.assertAlmostEqual(metrics.diameter.one_minus_fvu, 0.5)
        self.assertEqual(metrics.color.error_rate, 0.5)
        self.assertAlmostEqual(metrics.color.one_minus_fvu, -1.0)

    def test_single_class_color_has_no_fvu(self) -> None:
        records = {
            "TRL_gt": [2.0],
            "TRL_pred": [2.0],
            "dia_gt": [1.0],
            "dia_pred": [1.0],
            "color_gt": [1],
            "color_pred": [1],
        }

        metrics = compute_roots_per_image_metrics(records)

        self.assertEqual(metrics.color.error_rate, 0.0)
        self.assertIsNone(metrics.color.one_minus_fvu)

    def test_empty_matched_records_are_supported(self) -> None:
        records = {
            "TRL_gt": [-1.0],
            "TRL_pred": [7.0],
            "dia_gt": [-1.0],
            "dia_pred": [7.0],
            "color_gt": [-1],
            "color_pred": [1],
        }

        metrics = compute_roots_per_image_metrics(records)

        self.assertEqual(metrics.color.evaluated_samples, 0)
        self.assertIsNone(metrics.color.error_rate)


if __name__ == "__main__":
    unittest.main()
