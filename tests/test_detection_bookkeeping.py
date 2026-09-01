"""Characterization tests for detection-result bookkeeping."""

import unittest

import numpy as np

from legonet import config
from legonet.eval.detection_bookkeeping import record_detection_bookkeeping
from legonet.eval.evaluation_state import initiate_global_dicts


class DetectionBookkeepingTests(unittest.TestCase):
    """Verify aligned GT values and evaluation-state counters."""

    def setUp(self) -> None:
        self.original_type = config.Detect_and_Estimate.type

    def tearDown(self) -> None:
        config.Detect_and_Estimate.type = self.original_type

    def test_attribute_matches_restore_original_prediction_order(self) -> None:
        config.Detect_and_Estimate.type = "per_object_attributes"
        state = initiate_global_dicts(initiate=True)
        initiate_global_dicts(state, "roots.jpg")
        predicted_boxes = [
            [0.0, 0.0, 10.0, 10.0, 0.8],
            [0.0, 0.0, 10.0, 10.0, 0.9],
        ]
        annotated_boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0, 0.0, 42.0]]
        )
        gt_values = np.array([[1.0, 0.0, 42.0, 12.0, 0.5]])

        result = record_detection_bookkeeping(
            state,
            "roots.jpg",
            predicted_boxes,
            annotated_boxes,
            gt_values,
            attributes=True,
            counting=False,
            iou_threshold=0.5,
        )

        self.assertEqual(result.length_ground_truth, [-1, 12.0])
        self.assertEqual(result.diameter_ground_truth, [-1, 0.5])
        self.assertEqual(result.color_ground_truth, [-1, 1.0])
        self.assertEqual(result.matched_predictions, [False, True])
        self.assertEqual(state["found_orig_objects"], 1)
        self.assertEqual(state["FP"], 1)
        self.assertEqual(
            state["detections_data_any_crop"]["roots.jpg"]["label"], [0, 1]
        )
        self.assertEqual(state["not_found_gt"]["roots.jpg"]["label"], [])

    def test_counting_records_false_positive_and_missed_gt(self) -> None:
        config.Detect_and_Estimate.type = "per_object_counting"
        state = initiate_global_dicts(initiate=True)
        initiate_global_dicts(state, "grapes.jpg")

        result = record_detection_bookkeeping(
            state,
            "grapes.jpg",
            [[0.0, 0.0, 10.0, 10.0, 0.7]],
            np.empty((0, 6)),
            np.array([[4.0, 0.0, 7.0]]),
            attributes=False,
            counting=True,
            iou_threshold=0.5,
        )

        self.assertEqual(result.count_ground_truth, [-1])
        self.assertIsNone(result.max_overlaps)
        self.assertEqual(result.matched_predictions, [False])
        self.assertEqual(state["FP"], 1)
        self.assertEqual(
            state["detections_data_any_crop"]["grapes.jpg"]["score"],
            [0.7],
        )
        self.assertEqual(
            state["not_found_gt"]["grapes.jpg"]["gt_count"], [4.0]
        )
        self.assertEqual(state["not_found_gt"]["grapes.jpg"]["pred"], [-1])


if __name__ == "__main__":
    unittest.main()
