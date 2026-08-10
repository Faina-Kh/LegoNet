"""Tests for detection-evaluation input preparation."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from legonet.eval.detection_inputs import prepare_detection_evaluation_inputs


class DetectionInputsTests(unittest.TestCase):
    def test_prepares_prediction_rows_and_original_scale_gt(self):
        state = {"all_detections": [], "all_annotations": []}
        dataset = SimpleNamespace(
            image_data_points_location={"image.jpg": [{"x": 2, "y": 3}]}
        )
        boxes = torch.tensor([[[2.0, 4.0, 10.0, 12.0]]])

        result = prepare_detection_evaluation_inputs(
            state,
            dataset,
            {"bbox_annot": boxes},
            "image.jpg",
            np.array([[1.0, 2.0, 5.0, 6.0]]),
            np.array([0.9]),
            np.array([2.0]),
            have_gt=True,
            has_ground_truth_boxes=True,
        )

        np.testing.assert_allclose(result.detections[0], [1, 2, 5, 6, 0.9])
        np.testing.assert_allclose(result.original_scale_annotations[0], [1, 2, 5, 6])
        self.assertEqual(result.point_annotations, [{"x": 2, "y": 3}])
        self.assertEqual(len(state["all_detections"]), 1)
        self.assertEqual(len(state["all_annotations"]), 1)

    def test_without_gt_records_only_predictions(self):
        state = {"all_detections": [], "all_annotations": []}
        dataset = SimpleNamespace(image_data_points_location={})

        result = prepare_detection_evaluation_inputs(
            state,
            dataset,
            {},
            "image.jpg",
            np.array([[1.0, 2.0, 5.0, 6.0]]),
            np.array([0.8]),
            np.array([1.0]),
            have_gt=False,
            has_ground_truth_boxes=False,
        )

        self.assertEqual(result.point_annotations, [])
        self.assertEqual(result.ground_truth_boxes, [])
        self.assertEqual(state["all_annotations"], [])


if __name__ == "__main__":
    unittest.main()
