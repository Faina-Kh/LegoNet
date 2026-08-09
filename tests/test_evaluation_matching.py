"""Direct tests for the reusable IoU matching module."""

import unittest

import numpy as np
import torch

from legonet import config
from legonet.eval.matching import (
    assign_detection_to_gt,
    choose_boxes_by_iou_and_precision,
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

    def test_model_box_assignment_preserves_legacy_gt_ids(self) -> None:
        original_device = config.General.device
        original_threshold = config.Detection.iou_threshold
        config.General.device = "cpu"
        config.Detection.iou_threshold = 0.5
        try:
            assignments = choose_boxes_by_iou_and_precision(
                detections=torch.tensor(
                    [
                        [0.0, 0.0, 10.0, 10.0],
                        [0.0, 0.0, 10.0, 10.0],
                    ]
                ),
                annotations=torch.tensor(
                    [[[0.0, 0.0, 10.0, 10.0, 0.0, 42.0]]]
                ),
                detection_scores=torch.tensor([0.9, 0.8]),
            )
        finally:
            config.General.device = original_device
            config.Detection.iou_threshold = original_threshold

        self.assertEqual([item[0, -1].item() for item in assignments], [42.0, -1.0])


if __name__ == "__main__":
    unittest.main()
