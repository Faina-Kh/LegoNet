"""Tests for post-loop per-object metric aggregation."""

import unittest

from legonet.eval.metric_aggregation import aggregate_post_loop_metrics


class MetricAggregationTests(unittest.TestCase):
    def test_counting_aggregation_flattens_images(self):
        state = {
            "all_crops_GT_counts": [[2], [3]],
            "all_predicted_counts": [[2], [4]],
            "all_orig_GT_counts": [[2], [3]],
            "orig_rel_error": [0.0, 1 / 3],
            "found_orig_objects": 2,
            "FP": 1,
        }

        result = aggregate_post_loop_metrics(state, attributes=False)

        self.assertAlmostEqual(result.count_mae, 0.5)
        self.assertAlmostEqual(result.count_agreement, 0.5)
        self.assertAlmostEqual(result.precision_detection, 2 / 3)
        self.assertEqual(result.crop_count_metrics["num_positive"], 2)

    def test_attribute_aggregation_keeps_color_classification(self):
        state = {
            "all_predicted_lengths": [[2.0, 4.0]],
            "all_orig_GT_lengths": [[1.0, 4.0]],
            "all_predicted_dia": [[0.2, 0.5]],
            "all_orig_GT_dia": [[0.1, 0.5]],
            "all_predicted_color": [[1, 0]],
            "all_orig_GT_color": [[1, 1]],
            "orig_rel_error_length": [1.0, 0.0],
            "orig_rel_error_dia": [1.0, 0.0],
            "all_data_gt_color": [1, 1],
            "found_orig_objects": 2,
            "FP": 0,
        }

        result = aggregate_post_loop_metrics(state, attributes=True)

        self.assertAlmostEqual(result.length_mae, 0.5)
        self.assertAlmostEqual(result.diameter_mae, 0.05)
        self.assertAlmostEqual(result.color_metrics.accuracy, 0.5)

    def test_counting_aggregation_handles_no_matched_gt(self):
        state = {
            "all_crops_GT_counts": [[0]],
            "all_predicted_counts": [[1]],
            "all_orig_GT_counts": [[-1]],
            "orig_rel_error": [],
            "found_orig_objects": 0,
            "FP": 1,
        }

        result = aggregate_post_loop_metrics(state, attributes=False)

        self.assertFalse(result.has_original_gt)
        self.assertEqual(result.count_mae, -1)
        self.assertEqual(result.precision_detection, 0)

    def test_attribute_aggregation_handles_no_matched_gt(self):
        state = {
            "all_predicted_lengths": [],
            "all_orig_GT_lengths": [],
            "all_predicted_dia": [],
            "all_orig_GT_dia": [],
            "all_predicted_color": [],
            "all_orig_GT_color": [],
            "orig_rel_error_length": [],
            "orig_rel_error_dia": [],
            "all_data_gt_color": [],
            "found_orig_objects": 0,
            "FP": 0,
        }

        result = aggregate_post_loop_metrics(state, attributes=True)

        self.assertFalse(result.had_predictions)
        self.assertFalse(result.has_original_gt)
        self.assertEqual(result.length_relative_error, 100000)


if __name__ == "__main__":
    unittest.main()
