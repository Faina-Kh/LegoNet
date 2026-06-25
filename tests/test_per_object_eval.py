"""Tests for per-object evaluation matching helpers."""

import unittest

import numpy as np

from legonet.eval import perObject_eval


class PerObjectEvaluationTests(unittest.TestCase):
    """Characterize bbox-to-GT matching used before attribute evaluation."""

    def test_assign_detection_matches_unclaimed_gt_above_threshold(self):
        """A high-IoU detection is matched when its GT box is still unclaimed."""
        detection = np.array([0.0, 0.0, 10.0, 10.0, 0.9])
        annotations = np.array([[0.0, 0.0, 10.0, 10.0, 3.0, 42.0]])

        assigned, overlap, is_match = perObject_eval._assign_detection_to_gt(
            detection,
            annotations,
            detected_annotations=[],
            iou_threshold=0.5,
        )

        self.assertEqual(assigned, 0)
        self.assertAlmostEqual(overlap, 1.0)
        self.assertTrue(is_match)

    def test_assign_detection_rejects_duplicate_gt_match(self):
        """A second high-IoU detection for an already matched GT is not a new match."""
        detection = np.array([0.0, 0.0, 10.0, 10.0, 0.8])
        annotations = np.array([[0.0, 0.0, 10.0, 10.0, 3.0, 42.0]])

        assigned, overlap, is_match = perObject_eval._assign_detection_to_gt(
            detection,
            annotations,
            detected_annotations=[0],
            iou_threshold=0.5,
        )

        self.assertEqual(assigned, 0)
        self.assertAlmostEqual(overlap, 1.0)
        self.assertFalse(is_match)

    def test_assign_detection_rejects_low_iou_prediction(self):
        """A low-overlap detection is not a true per-object evaluation target."""
        detection = np.array([20.0, 20.0, 30.0, 30.0, 0.7])
        annotations = np.array([[0.0, 0.0, 10.0, 10.0, 3.0, 42.0]])

        _, overlap, is_match = perObject_eval._assign_detection_to_gt(
            detection,
            annotations,
            detected_annotations=[],
            iou_threshold=0.5,
        )

        self.assertAlmostEqual(overlap, 0.0)
        self.assertFalse(is_match)


if __name__ == "__main__":
    unittest.main()
