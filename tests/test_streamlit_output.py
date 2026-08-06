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

    def test_returns_empty_text_before_metrics_are_available(self) -> None:
        self.assertEqual(extract_evaluation_summary("loading model\n"), "")


if __name__ == "__main__":
    unittest.main()
