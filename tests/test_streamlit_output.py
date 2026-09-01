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

    def test_includes_keypoint_metrics_in_attribute_summary(self) -> None:
        output = """
image.jpg: avg_gt_dia: 0.5, avg_pred_dia: 0.4
Avg of per image rel_error of TRL:0.1000 | 1-FVU: 0.9000
Avg of per image rel_error of diameter:0.2000 | 1-FVU: 0.8000
Avg of per image absolute error of color: 0.3000 | 1-FVU: 0.7000
color_classes: ('non_white', 'white') | color_error_rate: 0.125 | color_1-FVU: 0.600 | color_confusion_matrix: ((7, 1), (0, 8))
Keypoint detection evaluation
mAP: 0.600
"""

        summary = extract_evaluation_summary(output)

        self.assertNotIn("image.jpg", summary)
        self.assertEqual(len(summary.splitlines()), 7)
        self.assertIn("absolute error of color", summary)
        self.assertIn(
            "color_error_rate: 0.125 | color_1-FVU: 0.600",
            summary,
        )
        self.assertNotIn("color_classes", summary)
        self.assertNotIn("color_confusion_matrix", summary)
        self.assertIn("mAP: 0.600", summary)
        self.assertIn("Keypoint detection evaluation", summary)

    def test_keeps_only_latest_keypoint_summary(self) -> None:
        output = """
Keypoint detection evaluation
mAP: 0.400
Keypoint detection evaluation
mAP: 0.700
"""

        summary = extract_evaluation_summary(output)

        self.assertNotIn("0.400", summary)
        self.assertIn("mAP: 0.700", summary)
        self.assertEqual(summary.count("Keypoint detection evaluation"), 1)

    def test_extracts_counting_summary(self) -> None:
        output = """
Evaluation Summary - per-object counting for IoU-matched GT boxes
orig_avg_abs_count_diff: 0.500 | orig_count_agreement: 0.750 | orig_MSE: 0.400 | orig_avg_relative_error: 0.100 | orig_1-FVU: 0.800000
"""

        summary = extract_evaluation_summary(output)

        self.assertIn("Evaluation Summary - per-object counting", summary)
        self.assertIn("orig_avg_abs_count_diff: 0.500", summary)

    def test_keeps_only_latest_repeated_summary_type(self) -> None:
        output = """
Evaluation Summary - per-object counting for IoU-matched GT boxes
orig_avg_abs_count_diff: 2.000 | orig_avg_relative_error: 0.500
Evaluation Summary - per-object counting for IoU-matched GT boxes
orig_avg_abs_count_diff: 1.000 | orig_avg_relative_error: 0.250
"""

        summary = extract_evaluation_summary(output)

        self.assertNotIn("2.000", summary)
        self.assertIn("orig_avg_abs_count_diff: 1.000", summary)
        self.assertEqual(summary.count("Evaluation Summary -"), 1)

    def test_extracts_per_image_attribute_one_minus_fvu(self) -> None:
        output = (
            "Abs_value_Diff: 0.250 | MSE 0.125 | "
            "MRD (Relative Error for gt>0): 0.100 | 1-FVU: 0.750\n"
        )

        summary = extract_evaluation_summary(output)

        self.assertEqual(summary, output.strip())

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

    def test_appends_runtime_per_image_summary_without_prior_print(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.txt"
            results_path.write_text("ordinary output\n", encoding="utf-8")

            section = append_evaluation_summary(
                str(results_path),
                "Evaluation Summary - per-image estimation\nMRD: 0.100",
            )

        self.assertIn("Evaluation Summary - per-image estimation", section)
        self.assertIn("MRD: 0.100", section)


if __name__ == "__main__":
    unittest.main()
