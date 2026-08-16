"""Tests for compact Streamlit evaluation output."""

import unittest
import tempfile
from pathlib import Path

from legonet.streamlit_output import (
    append_evaluation_summary,
    extract_evaluation_summary,
    separate_execution_time,
)


class StreamlitOutputTests(unittest.TestCase):
    """Verify that verbose per-image rows do not hide aggregate metrics."""

    def test_excludes_keypoint_metrics_from_attribute_summary(self) -> None:
        output = """
image.jpg: avg_gt_dia: 0.5, avg_pred_dia: 0.4
Avg of per image rel_error of TRL:0.1000 | 1-FVU: 0.9000
Avg of per image rel_error of diameter:0.2000 | 1-FVU: 0.8000
Avg of per image absolute error of color: 0.3000 | 1-FVU: 0.7000
color_classes: ('non_white', 'white') | color_error_rate: 0.125 | color_1-FVU: 0.600 | color_confusion_matrix: ((7, 1), (0, 8))
Evaluation Summary - keypoint detection
mAP: 0.600 | recall: 0.500 | precision: 0.400
"""

        summary = extract_evaluation_summary(output)

        self.assertNotIn("image.jpg", summary)
        self.assertEqual(len(summary.splitlines()), 4)
        self.assertIn("absolute error of color", summary)
        self.assertIn(
            "color_error_rate: 0.125 | color_1-FVU: 0.600",
            summary,
        )
        self.assertNotIn("color_classes", summary)
        self.assertNotIn("color_confusion_matrix", summary)
        self.assertNotIn("mAP: 0.600", summary)
        self.assertNotIn("Evaluation Summary - keypoint detection", summary)

    def test_extracts_counting_summary(self) -> None:
        output = """
Evaluation Summary - per-object counting for IoU-matched GT boxes
orig_avg_abs_count_diff: 0.500 | orig_count_agreement: 0.750 | orig_MSE: 0.400 | orig_avg_relative_error: 0.100 | orig_1-FVU: 0.800000
"""

        summary = extract_evaluation_summary(output)

        self.assertIn("Evaluation Summary - per-object counting", summary)
        self.assertIn("orig_avg_abs_count_diff: 0.500", summary)

    def test_returns_empty_text_before_metrics_are_available(self) -> None:
        self.assertEqual(extract_evaluation_summary("loading model\n"), "")

    def test_separates_execution_time_from_streamed_output(self) -> None:
        output, execution_time = separate_execution_time(
            "metric output\nExecution time in minutes: 1.624\n"
        )

        self.assertEqual(output, "metric output\n")
        self.assertEqual(execution_time, "Execution time in minutes: 1.624")

    def test_appends_consolidated_summary_to_text_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.txt"
            results_path.write_text(
                "Evaluation Summary - per-object attributes\n"
                "orig_avg_abs_TRL_diff: 0.100 | orig_MSE_TRL: 0.020\n",
                encoding="utf-8",
            )

            section = append_evaluation_summary(str(results_path))
            saved_output = results_path.read_text(encoding="utf-8")

        self.assertIn("\nEvaluation Summary\n", section)
        self.assertIn("orig_avg_abs_TRL_diff: 0.100", section)
        self.assertTrue(saved_output.endswith(section))


if __name__ == "__main__":
    unittest.main()
