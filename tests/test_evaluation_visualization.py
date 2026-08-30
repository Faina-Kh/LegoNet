"""Tests for optional per-object evaluation image artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageFont

from legonet.eval import KP_detection_eval
from legonet.eval.KP_detection_eval import visualize_KeyPointsHeatmaps
from legonet.eval.visualization import (
    _matched_gt_box,
    save_detection_overview,
    save_keypoint_heatmap,
    save_object_visualizations,
    scaled_box,
)


class EvaluationVisualizationTests(TestCase):
    """Verify visualization helpers without running model evaluation."""

    def test_heatmap_count_values_are_formatted_as_integers(self) -> None:
        self.assertEqual(KP_detection_eval._format_heatmap_value(12.0, "count"), "12")
        self.assertEqual(KP_detection_eval._format_heatmap_value(12.6, "count"), "13")

    def test_heatmap_length_values_keep_decimal_precision(self) -> None:
        self.assertEqual(KP_detection_eval._format_heatmap_value(12.56, "length"), "12.56")

    def test_point_evaluation_applies_historical_candidate_cutoff(self) -> None:
        predicted_map = torch.tensor([[0.01, 0.03], [0.0, 0.0]])
        ground_truth_map = np.array([[0.0, 1.0], [0.0, 0.0]])

        targets, scores = KP_detection_eval.points_detection_t_p(
            predicted_map, ground_truth_map
        )

        self.assertEqual(targets, [1])
        self.assertAlmostEqual(scores[0], 0.03, places=6)

    def test_empty_gt_map_records_candidates_as_false_positives(self) -> None:
        predicted_map = torch.tensor([[0.01, 0.03], [0.0, 0.0]])
        ground_truth_map = np.zeros((2, 2))

        targets, scores = KP_detection_eval.points_detection_t_p(
            predicted_map, ground_truth_map
        )

        self.assertEqual(targets, [0])
        self.assertAlmostEqual(scores[0], 0.03, places=6)

    def test_local_maximum_protocol_keeps_peaks_below_fixed_cutoff(self) -> None:
        predicted_map = torch.tensor(
            [[0.001, 0.005, 0.001], [0.002, 0.010, 0.002], [0.001, 0.003, 0.001]]
        )
        ground_truth_map = np.zeros((3, 3))

        targets, scores = KP_detection_eval.points_detection_t_p(
            predicted_map,
            ground_truth_map,
            candidate_threshold=None,
            local_maxima_only=True,
        )

        self.assertEqual(targets, [0])
        self.assertAlmostEqual(scores[0], 0.01, places=6)

    def test_scaled_box_uses_image_scale(self) -> None:
        self.assertEqual(
            scaled_box([10, 20, 30, 40], [2]),
            (5.0, 10.0, 15.0, 20.0),
        )

    @patch(
        "legonet.eval.KP_detection_eval.ImageFont.truetype",
        return_value=ImageFont.load_default(),
    )
    def test_prediction_heatmap_is_not_deleted(self, _font) -> None:
        with TemporaryDirectory() as temporary_directory:
            original_imsave = KP_detection_eval.plt.imsave
            with patch(
                "legonet.eval.KP_detection_eval.plt.imsave",
                side_effect=original_imsave,
            ) as imsave:
                visualize_KeyPointsHeatmaps(
                    np.ones((4, 4), dtype=float),
                    None,
                    "sample",
                    "sample_map_5",
                    Image.new("RGB", (16, 16), "white"),
                    temporary_directory,
                    heatmap_vmax=1.0,
                )

            output = Path(temporary_directory)
            self.assertTrue((output / "sample_map_5_Predicted.png").is_file())
            self.assertFalse(
                (output / "sample_map_5_predicted_map_tmp.png").exists()
            )
            self.assertEqual(imsave.call_args.kwargs["vmin"], 0.0)
            self.assertEqual(imsave.call_args.kwargs["vmax"], 1.0)

    @patch(
        "legonet.eval.KP_detection_eval.ImageFont.truetype",
        return_value=ImageFont.load_default(),
    )
    def test_gt_and_prediction_heatmaps_use_the_same_fixed_scale(self, _font) -> None:
        with TemporaryDirectory() as temporary_directory:
            original_imsave = KP_detection_eval.plt.imsave
            with patch(
                "legonet.eval.KP_detection_eval.plt.imsave",
                side_effect=original_imsave,
            ) as imsave:
                visualize_KeyPointsHeatmaps(
                    np.full((4, 4), 0.1, dtype=float),
                    np.ones((4, 4), dtype=float),
                    "sample",
                    "sample_map_5",
                    Image.new("RGB", (16, 16), "white"),
                    temporary_directory,
                    heatmap_vmax=1.0,
                )

            self.assertEqual(imsave.call_count, 2)
            for call in imsave.call_args_list:
                self.assertEqual(call.kwargs["vmin"], 0.0)
                self.assertEqual(call.kwargs["vmax"], 1.0)

    def test_object_visualizations_save_overlay_and_crop(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            image_path = output_path / "source.jpg"
            Image.new("RGB", (32, 32), "white").save(image_path)

            crop_image = save_object_visualizations(
                image_path=str(image_path),
                image_name="sample.jpg",
                crop_index=0,
                bbox_crop=np.ones((3, 16, 16), dtype=np.float32),
                predicted_box=[4, 4, 24, 24],
                scale=[1],
                gt_boxes=np.asarray([[5, 5, 20, 20, 0, 1]], dtype=float),
                gt_box_id=1,
                roots_attributes=True,
                image_points=[{"x": 10, "y": 10}],
                crop_points=[{"x": 8, "y": 8}],
                has_positive_target=True,
                predicted_boxes_path=str(output_path),
                crops_path=str(output_path),
                line_width=1,
                unnormalize=lambda crop: crop,
            )

            self.assertEqual(crop_image.size, (16, 16))
            self.assertTrue(
                (output_path / "sample_predicted_BBOX_0.png").is_file()
            )
            saved_crop_path = output_path / "sample_crop_0.png"
            self.assertTrue(saved_crop_path.is_file())
            saved_crop = Image.open(saved_crop_path)
            self.assertEqual(saved_crop.getpixel((8, 8)), (0, 0, 0))
            self.assertNotEqual(crop_image.getpixel((8, 8)), (0, 0, 0))

    def test_roots_matched_box_uses_bbox_id_not_row_position(self) -> None:
        gt_boxes = np.asarray(
            [
                [4, 6, 20, 24, 0, 0],
                [40, 60, 80, 100, 0, 7],
            ],
            dtype=float,
        )

        matched_box = _matched_gt_box(
            gt_boxes,
            gt_box_id=0,
            scale=[2],
            roots_attributes=True,
        )

        self.assertEqual(matched_box, (2.0, 3.0, 10.0, 12.0))

    def test_crop_is_saved_when_individual_box_overlay_is_disabled(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            image_path = output_path / "source.png"
            Image.new("RGB", (32, 32), "white").save(image_path)

            save_object_visualizations(
                image_path=str(image_path),
                image_name="sample.png",
                crop_index=0,
                bbox_crop=np.ones((3, 16, 16), dtype=np.float32),
                predicted_box=[4, 4, 24, 24],
                scale=[1],
                gt_boxes=np.empty((0, 6)),
                gt_box_id=-1,
                roots_attributes=False,
                image_points=[],
                crop_points=[],
                has_positive_target=False,
                predicted_boxes_path=str(output_path),
                crops_path=str(output_path),
                line_width=1,
                unnormalize=lambda crop: crop,
                save_predicted_box_overlay=False,
            )

            self.assertTrue((output_path / "sample_crop_0.png").is_file())
            self.assertFalse(
                (output_path / "sample_predicted_BBOX_0.png").exists()
            )

    def test_detection_overview_saves_gt_and_prediction_images(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            gt_path = output_path / "gt"
            gt_path.mkdir()
            image_path = output_path / "source.jpg"
            Image.new("RGB", (32, 32), "white").save(image_path)

            save_detection_overview(
                image_path=str(image_path),
                image_name="sample.jpg",
                predicted_boxes=np.asarray([[4, 4, 24, 24]], dtype=float),
                gt_boxes=np.asarray([[8, 8, 20, 20]], dtype=float),
                point_annotations=[{"x": 12, "y": 12}],
                scale=[1],
                have_gt=True,
                draw_path=str(output_path),
                gt_path=str(gt_path),
                line_width=1,
                point_radius=2,
                draw_detection_overview=True,
                draw_gt_only=True,
            )

            self.assertTrue((output_path / "sample.png").is_file())
            self.assertTrue((gt_path / "sample_all_Boxes.png").is_file())
            self.assertTrue((gt_path / "sample_gt_box_0.png").is_file())

    def test_detection_overview_does_not_require_gt_folder_when_disabled(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            image_path = output_path / "source.jpg"
            Image.new("RGB", (32, 32), "white").save(image_path)

            save_detection_overview(
                image_path=str(image_path),
                image_name="sample.jpg",
                predicted_boxes=np.asarray([[4, 4, 24, 24]], dtype=float),
                gt_boxes=np.asarray([[8, 8, 20, 20]], dtype=float),
                point_annotations=[],
                scale=[1],
                have_gt=True,
                draw_path=str(output_path),
                gt_path=str(output_path / "not-created"),
                line_width=1,
                point_radius=2,
                draw_detection_overview=True,
                draw_gt_only=False,
            )

            self.assertTrue((output_path / "sample.png").is_file())
            self.assertFalse((output_path / "not-created").exists())

    def test_gt_overview_can_skip_individual_box_images(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            gt_path = output_path / "gt"
            gt_path.mkdir()
            image_path = output_path / "source.png"
            Image.new("RGB", (32, 32), "white").save(image_path)

            save_detection_overview(
                image_path=str(image_path),
                image_name="sample.png",
                predicted_boxes=[],
                gt_boxes=np.asarray([[8, 8, 20, 20]], dtype=float),
                point_annotations=[],
                scale=[1],
                have_gt=True,
                draw_path=str(output_path),
                gt_path=str(gt_path),
                line_width=1,
                point_radius=2,
                draw_detection_overview=False,
                draw_gt_only=True,
                draw_individual_objects=False,
            )

            self.assertTrue((gt_path / "sample_all_Boxes.png").is_file())
            self.assertFalse((gt_path / "sample_gt_box_0.png").exists())

    @patch("legonet.eval.visualization._visualize_keypoint_heatmaps")
    def test_keypoint_heatmap_uses_fifth_map_and_advances_index(
        self, visualize_heatmap
    ) -> None:
        point_maps = [
            [np.full((2, 2), index, dtype=float)] for index in range(6)
        ]
        predicted_maps = [
            [np.full((2, 2), index, dtype=float) for index in range(5)]
        ]
        crop_image = Image.new("RGB", (8, 8))

        next_index = save_keypoint_heatmap(
            image_name="sample.jpg",
            crop_index=0,
            crop_image=crop_image,
            point_maps=point_maps,
            predicted_maps=predicted_maps,
            maps_index=0,
            draw_maps=True,
            maps_path="maps",
            predicted_value=12.5,
            ground_truth_value=14.0,
            attribute_name="count",
            attribute_unit="",
        )

        self.assertEqual(next_index, 1)
        visualize_heatmap.assert_called_once()
        arguments = visualize_heatmap.call_args[0]
        np.testing.assert_array_equal(arguments[0], predicted_maps[0][4])
        np.testing.assert_array_equal(arguments[1], point_maps[5][0])
        self.assertEqual(arguments[3], "sample_crop_0_map_5")
        self.assertEqual(visualize_heatmap.call_args.kwargs["count_pred"], 12.5)
        self.assertEqual(visualize_heatmap.call_args.kwargs["count_GT"], 14.0)
        self.assertEqual(
            visualize_heatmap.call_args.kwargs["attribute_name"], "count"
        )
        self.assertEqual(visualize_heatmap.call_args.kwargs["attribute_unit"], "")

    @patch("legonet.eval.visualization._visualize_keypoint_heatmaps")
    def test_disabled_keypoint_drawing_still_advances_index(
        self, visualize_heatmap
    ) -> None:
        next_index = save_keypoint_heatmap(
            image_name="sample.jpg",
            crop_index=0,
            crop_image=Image.new("RGB", (8, 8)),
            point_maps=[[np.zeros((2, 2))] for _ in range(6)],
            predicted_maps=[[np.zeros((2, 2)) for _ in range(5)]],
            maps_index=0,
            draw_maps=False,
            maps_path="maps",
        )

        self.assertEqual(next_index, 1)
        visualize_heatmap.assert_not_called()

    @patch("legonet.eval.visualization._visualize_keypoint_heatmaps")
    def test_keypoint_heatmap_supports_prediction_only_rendering(
        self, visualize_heatmap
    ) -> None:
        predicted_maps = [
            [np.full((2, 2), index, dtype=float) for index in range(5)]
        ]

        save_keypoint_heatmap(
            image_name="sample.png",
            crop_index=0,
            crop_image=Image.new("RGB", (8, 8)),
            point_maps=None,
            predicted_maps=predicted_maps,
            maps_index=0,
            draw_maps=True,
            maps_path="maps",
        )

        arguments = visualize_heatmap.call_args[0]
        np.testing.assert_array_equal(arguments[0], predicted_maps[0][4])
        self.assertIsNone(arguments[1])
