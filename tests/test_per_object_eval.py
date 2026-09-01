"""Tests for per-object evaluation matching helpers."""

import unittest

import numpy as np
import torch

from legonet.eval import perObject_eval


class PerObjectEvaluationTests(unittest.TestCase):
    """Characterize bbox-to-GT matching used before attribute evaluation."""

    def test_detection_rescaling_does_not_mutate_cached_cpu_boxes(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        original_boxes = boxes.clone()
        detection_outputs = (
            torch.tensor([0.9]),
            torch.tensor([0]),
            boxes,
        )

        _, scaled_boxes = perObject_eval._get_detections(
            detection_outputs,
            np.array([0.5]),
        )

        np.testing.assert_allclose(
            scaled_boxes,
            np.array([[20.0, 40.0, 60.0, 80.0]]),
        )
        torch.testing.assert_close(boxes, original_boxes)

    def test_point_evaluation_requires_non_empty_matched_crop(self):
        self.assertFalse(
            perObject_eval._include_crop_in_point_evaluation(
                torch.zeros((2, 2)), matched_prediction=True
            )
        )
        self.assertTrue(
            perObject_eval._include_crop_in_point_evaluation(
                torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
                matched_prediction=True,
            )
        )
        self.assertFalse(
            perObject_eval._include_crop_in_point_evaluation(
                torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
                matched_prediction=False,
            )
        )

    def test_grape_point_evaluation_uses_raw_nonzero_map(self):
        raw_map = torch.tensor([[0.0, 0.01], [0.0, 0.0]])
        processed_map = torch.zeros((2, 2))
        ground_truth_map = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

        targets, scores = perObject_eval._evaluate_crop_keypoints(
            raw_map,
            processed_map,
            ground_truth_map,
            evaluates_attributes=False,
        )

        self.assertEqual(targets, [1])
        self.assertAlmostEqual(scores[0], 0.01, places=6)

    def test_root_point_evaluation_uses_raw_nonzero_map(self):
        raw_map = torch.tensor([[0.0, 0.01], [0.0, 0.0]])
        processed_map = torch.zeros((2, 2))
        ground_truth_map = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

        targets, scores = perObject_eval._evaluate_crop_keypoints(
            raw_map,
            processed_map,
            ground_truth_map,
            evaluates_attributes=True,
        )

        self.assertEqual(targets, [1])
        self.assertAlmostEqual(scores[0], 0.01, places=6)

    def test_grape_metric_point_map_uses_geometric_containment(self):
        """Without an object ID, contained points from any box are included."""
        points = [
            {"x": 15.0, "y": 25.0, "bbox_id": 999},
            {"x": 50.0, "y": 50.0, "bbox_id": 1},
        ]

        center_map = perObject_eval._geometric_point_centers_map(
            point_annotations=points,
            crop_box=[10.0, 20.0, 30.0, 40.0],
            image_scale=[1.0],
            output_shape=(80, 80),
            crop_size=(640, 640),
        )

        self.assertEqual(np.sum(center_map), 1.0)
        self.assertEqual(center_map[20, 20], 1.0)

    def test_root_metric_point_map_uses_only_matched_bbox_id(self):
        """Root crop metrics match the object-specific training target."""
        points = [
            {"x": 15.0, "y": 25.0, "bbox_id": 7},
            {"x": 16.0, "y": 26.0, "bbox_id": 8},
        ]

        center_map = perObject_eval._geometric_point_centers_map(
            point_annotations=points,
            crop_box=[10.0, 20.0, 30.0, 40.0],
            image_scale=[1.0],
            output_shape=(80, 80),
            crop_size=(640, 640),
            matched_bbox_id=7,
        )

        self.assertEqual(np.sum(center_map), 1.0)
        self.assertEqual(center_map[20, 20], 1.0)

    def test_metric_point_map_applies_image_scale_before_crop_projection(self):
        """Original-image point coordinates align with scaled detector boxes."""
        points = [{"x": 10.0, "y": 10.0, "bbox_id": -1}]

        center_map = perObject_eval._geometric_point_centers_map(
            point_annotations=points,
            crop_box=[10.0, 10.0, 30.0, 30.0],
            image_scale=[2.0],
            output_shape=(80, 80),
            crop_size=(640, 640),
        )

        self.assertEqual(np.sum(center_map), 1.0)
        self.assertEqual(center_map[40, 40], 1.0)

    def test_prepare_gt_boxes_keeps_all_boxes_and_matches_annotated_boxes(self):
        """GT preparation keeps all boxes while selecting boxes with annotations."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.0, 101.0],
                [20.0, 20.0, 30.0, 30.0, 0.0, 202.0],
            ]
        )
        counts = np.array(
            [
                [4.0, 0.0, 101.0],
                [7.0, 0.0, 202.0],
            ]
        )

        all_boxes, boxes_with_annotations, matched_counts = (
            perObject_eval._prepare_gt_boxes_for_attribute_eval(boxes, counts)
        )

        self.assertEqual(len(all_boxes), 2)
        self.assertEqual(len(boxes_with_annotations), 2)
        self.assertEqual(len(matched_counts), 2)
        self.assertEqual(matched_counts[0][2], 101.0)
        self.assertEqual(matched_counts[1][2], 202.0)

    def test_prepare_gt_boxes_excludes_unannotated_boxes_from_attribute_targets(self):
        """A GT box without point/count annotations stays in all boxes only."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.0, 101.0],
                [20.0, 20.0, 30.0, 30.0, 0.0, 202.0],
            ]
        )
        counts = np.array([[4.0, 0.0, 101.0]])

        all_boxes, boxes_with_annotations, matched_counts = (
            perObject_eval._prepare_gt_boxes_for_attribute_eval(boxes, counts)
        )

        self.assertEqual(len(all_boxes), 2)
        self.assertEqual(len(boxes_with_annotations), 1)
        self.assertEqual(len(matched_counts), 1)
        self.assertEqual(float(boxes_with_annotations[0][0, 5]), 101.0)

    def test_prepare_gt_boxes_handles_no_attribute_annotations(self):
        """When no GT boxes have annotations, only the all-box list is populated."""
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.0, 101.0],
                [20.0, 20.0, 30.0, 30.0, 0.0, 202.0],
            ]
        )
        counts = []

        all_boxes, boxes_with_annotations, matched_counts = (
            perObject_eval._prepare_gt_boxes_for_attribute_eval(boxes, counts)
        )

        self.assertEqual(len(all_boxes), 2)
        self.assertEqual(boxes_with_annotations, [])
        self.assertEqual(matched_counts, [])

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

    def test_match_detections_classifies_every_prediction_once(self):
        """Matching returns one classification for every predicted box."""
        detections = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [0.0, 0.0, 10.0, 10.0, 0.8],
                [20.0, 20.0, 30.0, 30.0, 0.7],
            ]
        )
        annotations = np.array([[0.0, 0.0, 10.0, 10.0, 0.0, 42.0]])

        matches = perObject_eval._match_detections_to_gt(
            detections,
            annotations,
            iou_threshold=0.5,
        )

        self.assertEqual(len(matches), len(detections))
        self.assertEqual([match[3] for match in matches], [True, False, False])

    def test_crop_count_metrics_exclude_empty_crops(self):
        """Crop accuracy uses positive GT crops and reports empty ones separately."""
        metrics = perObject_eval._compute_positive_crop_count_metrics(
            ground_truth_counts=[0, 2, 4],
            predicted_counts=[7, 1, 6],
        )

        self.assertEqual(metrics["num_total"], 3)
        self.assertEqual(metrics["num_positive"], 2)
        self.assertEqual(metrics["num_empty"], 1)
        self.assertAlmostEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["mse"], 2.5)
        self.assertAlmostEqual(metrics["mean_relative_error"], 0.5)
        self.assertAlmostEqual(metrics["exact_agreement"], 0.0)

    def test_crop_count_metrics_handle_no_positive_crops(self):
        """All-empty crop sets produce explicit unavailable metric values."""
        metrics = perObject_eval._compute_positive_crop_count_metrics(
            ground_truth_counts=[0, 0],
            predicted_counts=[2, 3],
        )

        self.assertEqual(metrics["num_positive"], 0)
        self.assertEqual(metrics["num_empty"], 2)
        self.assertEqual(metrics["mae"], -1.0)
        self.assertEqual(metrics["mean_relative_error"], -1.0)

    def test_crop_count_metrics_reject_misaligned_inputs(self):
        """Crop diagnostics fail clearly when predictions and GT do not align."""
        with self.assertRaises(ValueError):
            perObject_eval._compute_positive_crop_count_metrics([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
