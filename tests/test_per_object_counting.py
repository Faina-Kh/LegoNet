"""Tests for per-object counting matched-GT targets."""

import unittest

import torch

from legonet.models.model_per_object_counting import (
    _count_target_batch,
    _full_count_for_box,
)


class PerObjectCountingTargetTests(unittest.TestCase):
    """Verify that counting supervision follows matched GT box identifiers."""

    def test_full_count_is_selected_by_box_id(self):
        annotations = torch.tensor(
            [
                [12.0, 0.0, 101.0],
                [37.0, 0.0, 202.0],
            ]
        )

        count = _full_count_for_box(annotations, 202.0)

        self.assertEqual(count.item(), 37.0)

    def test_missing_box_count_fails_clearly(self):
        annotations = torch.tensor([[12.0, 0.0, 101.0]])

        with self.assertRaisesRegex(ValueError, "GT box 202"):
            _full_count_for_box(annotations, 202.0)

    def test_scalar_count_is_formatted_as_single_item_batch(self):
        """Regression loss receives the two-dimensional target it expects."""
        target = _count_target_batch(torch.tensor(37.0))

        self.assertEqual(tuple(target.shape), (1, 1))
        self.assertEqual(target.item(), 37.0)

if __name__ == "__main__":
    unittest.main()
