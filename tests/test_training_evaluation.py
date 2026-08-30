"""Tests for state-safe in-training evaluation helpers."""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from legonet import config


class TrainingEvaluationTests(unittest.TestCase):
    """Characterize evaluation results and mutable-state restoration."""

    @classmethod
    def setUpClass(cls):
        """Import evaluation helpers with heavyweight evaluators stubbed."""
        evaluation_package = types.ModuleType("legonet.eval")
        evaluation_package.detection_eval = mock.Mock()
        evaluation_package.perObject_eval = mock.Mock()
        sys.modules.pop("legonet.training_evaluation", None)
        with mock.patch.dict(sys.modules, {"legonet.eval": evaluation_package}):
            cls.evaluation = importlib.import_module("legonet.training_evaluation")

    def setUp(self):
        """Reset evaluator mocks and the shared IoU threshold."""
        self.evaluation.detection_eval.reset_mock()
        self.evaluation.perObject_eval.reset_mock()
        self.evaluation.detection_eval.evaluateMAP_simple.side_effect = None
        self.evaluation.perObject_eval.eval.side_effect = None
        config.Detection.iou_threshold = 0.5

    def test_detection_switches_model_to_evaluation_mode_first(self):
        """Detection metrics are never collected while the model is training."""
        model = mock.Mock()

        def detection_result(*args, **kwargs):
            self.assertTrue(model.eval.called)
            return 0.6, 0.7, 0.8

        self.evaluation.detection_eval.evaluateMAP_simple.side_effect = detection_result

        metrics = self.evaluation.evaluate_detection("dataset", "loader", "sampler", model)

        self.assertEqual(metrics.mean_average_precision, 0.6)
        self.assertEqual(metrics.precision, 0.7)
        self.assertEqual(metrics.recall, 0.8)

    def test_iou_sweep_averages_valid_results_and_restores_state(self):
        """The sweep ignores missing errors and restores dataset and IoU state."""
        training_dataset = object()
        validation_dataset = object()
        model = mock.Mock(dataset=training_dataset)
        self.evaluation.perObject_eval.eval.side_effect = [
            (0.2, 10, 7, 0.7, 0.8),
            [],
            (0.4, 10, 5, 0.5, 0.75),
        ]

        result = self.evaluation.evaluate_combined_iou_sweep(
            validation_dataset,
            "loader",
            "sampler",
            model,
            args=object(),
            iou_thresholds=[0.3, 0.5, 0.7],
        )

        self.assertAlmostEqual(result.average_relative_error, 0.3)
        self.assertEqual(result.measurements[0].matched_objects, 7)
        self.assertEqual(result.measurements[0].recall, 0.7)
        self.assertEqual(result.measurements[0].precision, 0.8)
        self.assertIsNone(result.measurements[1].matched_objects)
        self.assertIs(model.dataset, training_dataset)
        self.assertEqual(config.Detection.iou_threshold, 0.5)

    def test_empty_iou_sweep_has_no_average(self):
        """A sweep with no valid evaluator output cannot select a checkpoint."""
        model = mock.Mock(dataset=object())
        self.evaluation.perObject_eval.eval.side_effect = [[], (-1,)]

        result = self.evaluation.evaluate_combined_iou_sweep(
            object(),
            "loader",
            "sampler",
            model,
            args=object(),
            iou_thresholds=[0.3, 0.7],
        )

        self.assertIsNone(result.average_relative_error)

    def test_checkpoint_metrics_extract_relative_error_and_one_minus_fvu(self):
        """Silent training evaluation returns checkpoint-selection metrics."""
        training_dataset = object()
        model = mock.Mock(dataset=training_dataset)
        self.evaluation.perObject_eval.eval.return_value = SimpleNamespace(
            count_relative_error=0.2,
            count_fvu=0.35,
            length_relative_error=100000,
            length_mae=-1,
            length_mse=-1,
            length_fvu=-1,
            diameter_relative_error=-1,
            diameter_mae=-1,
            diameter_mse=-1,
            diameter_fvu=-1,
            color_metrics=None,
        )

        summary = self.evaluation.evaluate_per_object_checkpoint_metrics(
            object(),
            "loader",
            "sampler",
            model,
            args=SimpleNamespace(
                network_type="per_object_counting",
                checkpoint_attribute=None,
            ),
        )

        self.assertEqual(summary.metric_name, "count_relative_error")
        self.assertEqual(summary.metric_value, 0.2)
        self.assertEqual(summary.one_minus_fvu, 0.65)
        self.assertFalse(self.evaluation.perObject_eval.eval.call_args.kwargs["verbose"])
        self.assertTrue(
            self.evaluation.perObject_eval.eval.call_args.kwargs["return_metrics"]
        )
        self.assertIs(model.dataset, training_dataset)

    def test_selects_requested_continuous_attribute_metric(self):
        metrics = SimpleNamespace(
            count_relative_error=-1,
            count_fvu=-1,
            length_relative_error=0.2,
            length_mae=1.0,
            length_mse=2.0,
            length_fvu=0.25,
            diameter_relative_error=0.3,
            diameter_mae=0.4,
            diameter_mse=0.5,
            diameter_fvu=0.35,
            color_metrics=None,
        )

        summary = self.evaluation.select_checkpoint_metrics(
            metrics,
            network_type="per_object_attributes",
            requested_attribute="diameter",
        )

        self.assertEqual(summary.metric_name, "diameter_relative_error")
        self.assertEqual(summary.metric_value, 0.3)
        self.assertAlmostEqual(summary.one_minus_fvu, 0.65)

    def test_color_selection_always_uses_error_rate(self):
        color_metrics = SimpleNamespace(
            error_rate=0.1,
            accuracy=0.9,
            one_minus_fvu=0.7,
        )
        metrics = SimpleNamespace(
            count_relative_error=-1,
            count_fvu=-1,
            length_relative_error=0.2,
            length_mae=1.0,
            length_mse=2.0,
            length_fvu=0.25,
            diameter_relative_error=0.3,
            diameter_mae=0.4,
            diameter_mse=0.5,
            diameter_fvu=0.35,
            color_metrics=color_metrics,
        )

        summary = self.evaluation.select_checkpoint_metrics(
            metrics,
            network_type="per_object_attributes",
            requested_attribute="color",
        )

        self.assertEqual(summary.metric_name, "color_error_rate")
        self.assertEqual(summary.metric_value, 0.1)
        self.assertEqual(summary.one_minus_fvu, 0.7)

    def test_iou_sweep_restores_state_when_evaluation_fails(self):
        """Evaluator exceptions cannot leak validation state into training."""
        training_dataset = object()
        model = mock.Mock(dataset=training_dataset)
        self.evaluation.perObject_eval.eval.side_effect = RuntimeError("evaluation failed")

        with self.assertRaises(RuntimeError):
            self.evaluation.evaluate_combined_iou_sweep(
                object(),
                "loader",
                "sampler",
                model,
                args=object(),
                iou_thresholds=[0.3],
            )

        self.assertIs(model.dataset, training_dataset)
        self.assertEqual(config.Detection.iou_threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
