"""Tests for structured per-crop prediction preparation."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from legonet.eval.crop_predictions import prepare_crop_predictions


def _state(attribute_mode: bool) -> dict:
    record = (
        {"color_pred": [], "TRL_pred": [], "dia_pred": []}
        if attribute_mode
        else {"pred": []}
    )
    return {
        "crops_without_gt_points": 0,
        "detections_data_any_crop": {"image.jpg": record},
    }


class CropPredictionPreparationTests(unittest.TestCase):
    def test_prepares_counting_predictions_and_crop_targets(self) -> None:
        state = _state(attribute_mode=False)
        sample_annotations = {
            "points_annot": [torch.tensor([2.0, 0.0])],
        }

        result = prepare_crop_predictions(
            state=state,
            dataset=SimpleNamespace(),
            image_name="image.jpg",
            predicted_boxes=np.zeros((2, 4)),
            original_crop_boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
            estimation_outputs=[torch.tensor([1.6, 0.2])],
            sample_annotations=sample_annotations,
            scale=[1.0],
            evaluates_attributes=False,
            have_ground_truth=True,
            estimate_type="regression",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: np.zeros((2, 2)),
        )

        self.assertEqual(result.count_predictions, [2.0, 0.0])
        self.assertEqual(result.ground_truth_counts, [2.0, 0.0])
        self.assertEqual(len(result.adjusted_original_boxes), 2)
        self.assertEqual(state["crops_without_gt_points"], 1)
        self.assertEqual(
            state["detections_data_any_crop"]["image.jpg"]["pred"],
            [2.0, 0.0],
        )

    def test_decodes_attribute_outputs_into_structured_lists(self) -> None:
        state = _state(attribute_mode=True)

        result = prepare_crop_predictions(
            state=state,
            dataset=SimpleNamespace(),
            image_name="image.jpg",
            predicted_boxes=np.zeros((2, 4)),
            original_crop_boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
            estimation_outputs=[
                torch.tensor([[0.8, 4.0, 0.5], [0.2, 6.0, 0.7]])
            ],
            sample_annotations=None,
            scale=[1.0],
            evaluates_attributes=True,
            have_ground_truth=False,
            estimate_type="regression",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: np.zeros((2, 2)),
        )

        self.assertEqual(result.color_predictions, [1, 0])
        self.assertEqual(result.trl_predictions, [4.0, 6.0])
        self.assertAlmostEqual(result.diameter_predictions[0], 0.5)
        self.assertAlmostEqual(result.diameter_predictions[1], 0.7)
        record = state["detections_data_any_crop"]["image.jpg"]
        self.assertEqual(record["color_pred"], [1, 0])
        self.assertEqual(record["TRL_pred"], [4.0, 6.0])

    def test_prepares_keypoint_maps_with_aligned_batch_dimensions(self) -> None:
        state = _state(attribute_mode=False)
        maps = [torch.full((1, 2, 2), float(index)) for index in range(7)]
        estimation_outputs = [torch.tensor([1.0]), *maps]
        dataset = SimpleNamespace(
            image_data_points_location={"image.jpg": [{"x": 2, "y": 3}]}
        )

        result = prepare_crop_predictions(
            state=state,
            dataset=dataset,
            image_name="image.jpg",
            predicted_boxes=np.zeros((1, 4)),
            original_crop_boxes=[[1, 2, 3, 4]],
            estimation_outputs=estimation_outputs,
            sample_annotations={
                "points_annot": [torch.tensor([1.0])],
            },
            scale=[1.0],
            evaluates_attributes=False,
            have_ground_truth=True,
            estimate_type="withKeyPoints",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: np.ones((2, 2)),
        )

        self.assertEqual(tuple(result.predicted_detection_maps.shape), (1, 2, 2))
        self.assertEqual(
            tuple(result.ground_truth_detection_maps.shape),
            (1, 2, 2),
        )
        self.assertEqual(len(result.predicted_maps_to_draw), 1)
        self.assertEqual(len(result.predicted_maps_to_draw[0]), 5)
        torch.testing.assert_close(
            result.predicted_maps_to_draw[0][4], maps[6][0]
        )

    def test_legacy_keypoint_outputs_fall_back_to_last_raw_map(self) -> None:
        state = _state(attribute_mode=False)
        maps = [torch.full((1, 2, 2), float(index)) for index in range(6)]

        result = prepare_crop_predictions(
            state=state,
            dataset=SimpleNamespace(),
            image_name="image.jpg",
            predicted_boxes=np.zeros((1, 4)),
            original_crop_boxes=[[1, 2, 3, 4]],
            estimation_outputs=[torch.tensor([1.0]), *maps],
            sample_annotations={},
            scale=[1.0],
            evaluates_attributes=False,
            have_ground_truth=False,
            estimate_type="withKeyPoints",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: np.zeros((2, 2)),
        )

        torch.testing.assert_close(
            result.predicted_maps_to_draw[0][4], maps[5][0]
        )

    def test_prepares_keypoint_predictions_without_ground_truth(self) -> None:
        """No-GT inference must not access point-annotation dataset fields."""
        state = _state(attribute_mode=True)
        maps = [torch.full((1, 2, 2), float(index)) for index in range(7)]
        estimation_outputs = [torch.tensor([[0.8, 4.0, 0.5]]), *maps]

        result = prepare_crop_predictions(
            state=state,
            dataset=SimpleNamespace(),
            image_name="image.jpg",
            predicted_boxes=np.zeros((1, 4)),
            original_crop_boxes=[[1, 2, 3, 4]],
            estimation_outputs=estimation_outputs,
            sample_annotations={},
            scale=[1.0],
            evaluates_attributes=True,
            have_ground_truth=False,
            estimate_type="withKeyPoints",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: self.fail(
                "GT point-map construction should not run without GT"
            ),
        )

        self.assertEqual(result.ground_truth_detection_maps, [])
        self.assertEqual(tuple(result.predicted_detection_maps.shape), (1, 2, 2))
        self.assertEqual(result.trl_predictions, [4.0])

    def test_no_gt_counting_preserves_crop_images_for_visualization(self) -> None:
        state = _state(attribute_mode=False)
        sample_annotations = {"img": [object()]}

        result = prepare_crop_predictions(
            state=state,
            dataset=SimpleNamespace(),
            image_name="image.jpg",
            predicted_boxes=np.zeros((1, 4)),
            original_crop_boxes=[[1, 2, 3, 4]],
            estimation_outputs=[torch.tensor([2.0])],
            sample_annotations=sample_annotations,
            scale=[1.0],
            evaluates_attributes=False,
            have_ground_truth=False,
            estimate_type="regression",
            crop_size=(640, 640),
            point_center_map_builder=lambda *args: np.zeros((2, 2)),
        )

        self.assertIs(result.sample_annotations, sample_annotations)


if __name__ == "__main__":
    unittest.main()
