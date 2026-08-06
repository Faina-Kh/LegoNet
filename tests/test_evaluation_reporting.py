"""Tests for evaluation CSV artifact writers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from legonet.eval.reporting import (
    write_evaluation_artifacts,
    write_keypoint_precision_recall,
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
                _read_csv(output / "not_found_gt_count.csv")[1],
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
            "TRL_gt": [10], "TRL_pred": [9],
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


if __name__ == "__main__":
    unittest.main()
