"""Direct tests for the reusable IoU matching module."""

import unittest

import numpy as np

from legonet.eval.matching import (
    assign_detection_to_gt,
    match_detections_to_gt,
)


class EvaluationMatchingTests(unittest.TestCase):
    """Verify matching independently of the legacy evaluator module."""

    def test_empty_annotations_make_detection_a_false_positive(self) -> None:
        assigned, overlap, is_match = assign_detection_to_gt(
            np.array([0.0, 0.0, 10.0, 10.0, 0.9]),
            np.empty((0, 6)),
            detected_annotations=[],
            iou_threshold=0.5,
        )

        self.assertIsNone(assigned)
        self.assertEqual(overlap, -1.0)
        self.assertFalse(is_match)

    def test_greedy_matching_claims_each_gt_once(self) -> None:
        detections = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [0.0, 0.0, 10.0, 10.0, 0.8],
            ]
        )
        annotations = np.array([[0.0, 0.0, 10.0, 10.0, 0.0, 42.0]])

        matches = match_detections_to_gt(detections, annotations, 0.5)

        self.assertEqual([match[3] for match in matches], [True, False])


if __name__ == "__main__":
    unittest.main()
