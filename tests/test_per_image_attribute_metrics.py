"""Tests for metrics across image-level attribute aggregates."""

import unittest

from legonet.eval.per_image_attribute_metrics import (
    compute_per_image_attribute_metrics,
)


class PerImageAttributeMetricsTests(unittest.TestCase):
    """Verify that 1-FVU uses one aggregate observation per image."""

    def test_calculates_fvu_across_per_image_values(self) -> None:
        metrics = compute_per_image_attribute_metrics(
            trl_ground_truth=[10.0, 20.0, 30.0],
            trl_predictions=[10.0, 15.0, 35.0],
            diameter_ground_truth=[1.0, 2.0, 3.0],
            diameter_predictions=[1.0, 1.0, 4.0],
            color_ground_truth=[0.25, 0.5, 0.75],
            color_predictions=[0.25, 0.25, 1.0],
        )

        self.assertAlmostEqual(metrics.trl.one_minus_fvu, 0.75)
        self.assertAlmostEqual(metrics.diameter.one_minus_fvu, 0.0)
        self.assertAlmostEqual(metrics.color.one_minus_fvu, 0.0)

    def test_color_uses_image_means_not_object_classification_accuracy(self) -> None:
        metrics = compute_per_image_attribute_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [0.25, 0.75],
            [0.5, 0.5],
        )

        self.assertAlmostEqual(metrics.color.mean_absolute_error, 0.25)
        self.assertAlmostEqual(metrics.color.mean_relative_error, 2.0 / 3.0)
        self.assertEqual(metrics.color.one_minus_fvu, 0.0)

    def test_rejects_misaligned_values_for_an_attribute(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_image_attribute_metrics(
                [1.0],
                [],
                [1.0],
                [1.0],
                [0.0],
                [0.0],
            )


if __name__ == "__main__":
    unittest.main()
