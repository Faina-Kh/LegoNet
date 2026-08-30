"""Tests for evaluation CSV artifact writers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from legonet.eval.reporting import (
    format_attribute_summary,
    format_keypoint_summary,
    format_matching_diagnostics,
    format_prediction_aggregates,
    format_roots_per_image,
    write_evaluation_artifacts,
    write_keypoint_precision_recall,
    write_keypoint_summary,
)
from legonet.eval.per_image_attribute_metrics import (
    compute_per_image_attribute_metrics,
)
from legonet.eval.per_object_result import (
    ClassificationMetrics,
    ClassificationType,
)


def _read_csv(path: Path) -> list[list[str]]:
    """Read a CSV artifact as string rows."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.reader(input_file))


class EvaluationReportingTests(unittest.TestCase):
    """Verify exact public filenames, headers, rows, and placeholders."""

    def test_writes_counting_artifacts(self) -> None:
        state = {
            "detections_data_any_crop": {
                "a.jpg": {
                    "gt_count": [4], "pred": [5], "label": [1],
                    "score": [0.9], "max_overlap": [0.8],
                }
            },
            "not_found_gt": {
                "a.jpg": {
                    "gt_count": [7], "pred": [-1], "label": [1],
                    "score": [-1], "max_overlap": [-1],
                }
            },
            "no_predictions": {
                "empty.jpg": {
                    "gt_count": [], "pred": [], "label": [],
                    "score": [], "max_overlap": [],
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_artifacts(directory, state, attributes=False)
            output = Path(directory)

            self.assertEqual(
                _read_csv(output / "detections_data_any_crop.csv"),
                [
                    ["img", "crop", "gt_count", "pred_count", "label", "score", "max_overlap"],
                    ["a.jpg", "0", "4", "5", "1", "0.9", "0.8"],
                ],
            )
            self.assertEqual(
                _read_csv(output / "not_found_gt.csv")[1],
                ["a.jpg", "7", "-1", "1", "-1", "-1"],
            )
            self.assertEqual(
                _read_csv(output / "images_without_detections.csv")[1],
                ["empty.jpg", "0", "0", "-1", "-1", "-1"],
            )

    def test_writes_root_attribute_artifacts(self) -> None:
        record = {
            "color_gt": [1], "color_pred": [0], "label": [1],
            "score": [0.75], "max_overlap": [0.6],
            "length_gt": [10], "length_pred": [9],
            "dia_gt": [2], "dia_pred": [3],
        }
        empty_record = {key: [] for key in record}
        state = {
            "detections_data_any_crop": {"root.jpg": record},
            "not_found_gt": {"root.jpg": record},
            "no_predictions": {"empty.jpg": empty_record},
        }

        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_artifacts(directory, state, attributes=True)
            output = Path(directory)

            self.assertEqual(
                _read_csv(output / "detections_data_any_crop.csv")[1],
                ["root.jpg", "0", "1", "0", "1", "0.75", "0.6", "10", "9", "2", "3"],
            )
            self.assertEqual(
                _read_csv(output / "images_without_detections.csv")[1],
                ["empty.jpg", "-1", "-1", "-1", "-1", "-1", "0", "0", "0", "0"],
            )

    def test_no_gt_does_not_write_not_found_gt_artifact(self) -> None:
        state = {
            "detections_data_any_crop": {},
            "not_found_gt": {},
            "no_predictions": {},
        }

        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_artifacts(
                directory,
                state,
                attributes=True,
                have_gt=False,
            )
            output = Path(directory)

            self.assertFalse((output / "not_found_gt.csv").exists())
            self.assertTrue((output / "detections_data_any_crop.csv").exists())
            self.assertTrue((output / "images_without_detections.csv").exists())

    def test_writes_keypoint_precision_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_keypoint_precision_recall(directory, [0.2, 0.5], [1.0, 0.8])

            self.assertEqual(
                _read_csv(Path(directory) / "parts_recall_precision.csv"),
                [["recall", "precision"], ["0.2", "1.0"], ["0.5", "0.8"]],
            )

    def test_rejects_misaligned_keypoint_curves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_keypoint_precision_recall(directory, [0.2], [1.0, 0.8])

    def test_formats_and_writes_keypoint_summary(self) -> None:
        report = format_keypoint_summary(0.75)

        self.assertIn("Keypoint detection evaluation", report)
        self.assertIn("mAP: 0.750", report)
        self.assertNotIn("recall:", report)
        self.assertNotIn("precision:", report)

        with tempfile.TemporaryDirectory() as directory:
            write_keypoint_summary(directory, 0.75, 0.6, 0.8)

            self.assertEqual(
                _read_csv(Path(directory) / "keypoint_summary.csv"),
                [
                    ["mAP", "recall", "precision"],
                    ["0.75", "0.6", "0.8"],
                ],
            )

    def test_matching_diagnostics_state_the_evaluation_scope(self) -> None:
        report = format_matching_diagnostics(
            "all images, including empty images", 10, 6, 2
        )

        self.assertIn("all images, including empty images", report)
        self.assertIn("60.00% recall", report)
        self.assertIn("75.00%", report)

    def test_formats_no_gt_attribute_prediction_aggregates(self) -> None:
        report = format_prediction_aggregates(
            {"a.png": 0.5},
            trl_predictions={"a.png": 12.0},
            diameter_predictions={"a.png": 0.75},
            images_without_detections=("empty.png",),
        )

        self.assertIn("Per-image predicted aggregates", report)
        self.assertIn("predicted length sum (TRL): 12.00", report)
        self.assertIn("predicted diameter average: 0.75", report)
        self.assertIn("predicted color average: 0.50", report)
        self.assertIn("empty.png: no detected objects", report)

    def test_attribute_summary_reports_unavailable_color_metrics(self) -> None:
        color = ClassificationMetrics(
            classification_type=ClassificationType.BINARY,
            class_names=("non_white", "white"),
            confusion_matrix=((0, 0), (0, 0)),
            evaluated_samples=0,
            correct_predictions=0,
        )

        report = format_attribute_summary(
            1.0, 2.0, 0.1, 0.8,
            0.2, 0.3, 0.4, 0.5,
            color,
        )

        self.assertIn("orig_avg_relative_error_length: 0.100", report)
        self.assertIn("color_error_rate: n/a", report)
        self.assertIn("color_1-FVU: n/a", report)
        self.assertIn(
            "color_error_rate: n/a | color_1-FVU: n/a\n",
            report,
        )

    def test_roots_per_image_places_fvu_beside_each_error(self) -> None:
        trl_gt = {"a.jpg": 10.0, "b.jpg": 20.0}
        trl_pred = {"a.jpg": 9.0, "b.jpg": 18.0}
        diameter_gt = {"a.jpg": 1.0, "b.jpg": 2.0}
        diameter_pred = {"a.jpg": 1.0, "b.jpg": 1.5}
        color_gt = {"a.jpg": 0.25, "b.jpg": 0.75}
        color_pred = {"a.jpg": 0.5, "b.jpg": 0.5}
        metrics = compute_per_image_attribute_metrics(
            list(trl_gt.values()), list(trl_pred.values()),
            list(diameter_gt.values()), list(diameter_pred.values()),
            list(color_gt.values()), list(color_pred.values()),
        )

        report = format_roots_per_image(
            trl_gt,
            trl_pred,
            diameter_gt,
            diameter_pred,
            color_gt,
            color_pred,
            metrics,
        )

        self.assertIn("rel_error of TRL:", report)
        self.assertIn("rel_error of diameter:", report)
        self.assertIn("absolute error of color:", report)
        self.assertEqual(report.count("1-FVU:"), 3)
        self.assertNotIn("Per-image aggregate 1-FVU", report)


if __name__ == "__main__":
    unittest.main()
