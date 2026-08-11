"""Tests for task-neutral numerical metric helpers."""

import unittest

from legonet.eval.numeric_metrics import (
    sum_of_absolute_differences,
    sum_of_differences,
)


class NumericMetricTests(unittest.TestCase):
    def test_sums_signed_and_absolute_differences(self) -> None:
        ground_truth = [3.0, 5.0]
        predictions = [2.0, 7.0]

        self.assertEqual(sum_of_differences(ground_truth, predictions), -1.0)
        self.assertEqual(
            sum_of_absolute_differences(ground_truth, predictions),
            3.0,
        )

    def test_rejects_misaligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            sum_of_absolute_differences([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
