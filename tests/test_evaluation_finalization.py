"""Tests for post-aggregation evaluation finalization."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from legonet.eval.evaluation_finalization import (
    _print_detection_diagnostics,
    _print_verbose_details,
    finalize_evaluation,
    build_legacy_result,
)
from legonet.eval.metric_aggregation import PostLoopMetrics


class EvaluationFinalizationTests(unittest.TestCase):
    def test_print_to_files_emits_summary_when_verbose_is_false(self):
        state = {"gt_objects_withGTpoints": 5, "found_orig_objects": 4}
        metrics = PostLoopMetrics(
            had_predictions=True,
            has_original_gt=True,
            precision_detection=0.8,
            count_relative_error=0.1,
            count_mae=0.5,
            count_agreement=0.75,
            count_mse=0.4,
            count_fvu=0.2,
            crop_count_metrics={
                "num_positive": 4,
                "num_empty": 1,
                "num_total": 5,
                "mae": 0.5,
                "exact_agreement": 0.75,
                "mse": 0.4,
                "mean_relative_error": 0.1,
            },
        )
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            text_results = Path(directory) / "results.txt"
            with patch(
                "legonet.eval.evaluation_finalization.write_evaluation_artifacts"
            ):
                with redirect_stdout(output):
                    finalize_evaluation(
                        state,
                        metrics,
                        attributes=False,
                        have_gt=True,
                        predict_empty_image=False,
                        verbose=False,
                        print_to_files=True,
                        detection_metrics=None,
                        evaluate_points=False,
                        files_path=directory,
                        text_results_path=str(text_results),
                    )
            saved_output = text_results.read_text(encoding="utf-8")

        self.assertIn("Evaluation Summary - per-object counting", output.getvalue())
        self.assertIn(
            "Evaluation Summary - per-object counting",
            saved_output,
        )

    def test_counting_result_preserves_legacy_field_order(self):
        state = {"gt_objects_withGTpoints": 5, "found_orig_objects": 4}
        metrics = PostLoopMetrics(
            had_predictions=True,
            has_original_gt=True,
            precision_detection=0.8,
            count_relative_error=0.1,
            count_mae=0.5,
            count_agreement=0.75,
            count_mse=0.4,
            count_fvu=0.2,
        )

        result = build_legacy_result(
            state, metrics, attributes=False, detection_metrics=None
        )

        self.assertEqual(result, [0.1, 5, 4, 0.8, 0.8, 0.5, 0.75, 0.4, 0.8])

    def test_attribute_result_remains_training_metric_only(self):
        metrics = PostLoopMetrics(
            had_predictions=True,
            has_original_gt=True,
            precision_detection=1.0,
            trl_relative_error=0.25,
        )

        result = build_legacy_result(
            {}, metrics, attributes=True, detection_metrics=None
        )

        self.assertEqual(result, [0.25])

    def test_per_object_diagnostics_do_not_repeat_bbox_detection_stats(self):
        state = {
            "gt_objects_withGTpoints": 5,
            "found_orig_objects": 4,
            "FP": 1,
        }
        output = io.StringIO()

        with redirect_stdout(output):
            _print_detection_diagnostics(
                state,
                predict_empty_image=False,
                detection_metrics=[0.75, 0.8, 0.6],
            )

        report = output.getvalue()
        self.assertIn("Object matching diagnostics", report)
        self.assertNotIn("Bounding-box detection stats", report)
        self.assertNotIn("mAP:", report)

    def test_no_gt_prints_predictions_without_matching_diagnostics(self):
        state = {
            "per_im_pred_dict": {"image.png": 0.5},
            "TRL_per_im_pred_dict": {"image.png": 8.0},
            "dia_per_im_pred_dict": {"image.png": 0.4},
            "no_predictions": {},
        }
        output = io.StringIO()

        with redirect_stdout(output):
            _print_verbose_details(
                state,
                attributes=True,
                have_gt=False,
                predict_empty_image=True,
                detection_metrics=None,
            )

        report = output.getvalue()
        self.assertNotIn("Object matching diagnostics", report)
        self.assertIn("Per-image predicted aggregates", report)
        self.assertIn("predicted length sum (TRL): 8.00", report)


if __name__ == "__main__":
    unittest.main()
