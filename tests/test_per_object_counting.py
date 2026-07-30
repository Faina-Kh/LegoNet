"""Tests for per-object counting matched-GT targets."""

import unittest

import torch

from legonet.models.model_per_object_counting import (
    _count_target_batch,
    _full_count_for_box,
    _has_valid_detection_annotations,
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

    def test_padded_empty_bbox_annotations_are_not_valid(self):
        """The collater's all-minus-one row represents no bbox annotations."""
        annotations = torch.full((1, 6), -1.0)

        self.assertFalse(_has_valid_detection_annotations(annotations))

    def test_real_bbox_annotation_is_valid(self):
        """A nonnegative x1 identifies a real bbox row among padding rows."""
        annotations = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [2.0, 3.0, 10.0, 12.0, 0.0, 7.0],
            ]
        )

        self.assertTrue(_has_valid_detection_annotations(annotations))

if __name__ == "__main__":
    unittest.main()
