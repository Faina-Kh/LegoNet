"""Tests for compact Streamlit evaluation output."""

import unittest

from legonet.streamlit_output import extract_evaluation_summary


class StreamlitOutputTests(unittest.TestCase):
    """Verify that verbose per-image rows do not hide aggregate metrics."""

    def test_extracts_attribute_and_keypoint_summary_lines(self) -> None:
        output = """
image.jpg: avg_gt_dia: 0.5, avg_pred_dia: 0.4
Avg of per image rel_error of TRL:0.1000 | 1-FVU: 0.9000
Avg of per image rel_error of diameter:0.2000 | 1-FVU: 0.8000
Avg of per image absolute error of color: 0.3000 | 1-FVU: 0.7000
mAP: 0.600 | recall: 0.500 | precision: 0.400
"""

        summary = extract_evaluation_summary(output)

        self.assertNotIn("image.jpg", summary)
        self.assertEqual(len(summary.splitlines()), 4)
        self.assertIn("absolute error of color", summary)
        self.assertIn("mAP: 0.600", summary)

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


if __name__ == "__main__":
    unittest.main()
