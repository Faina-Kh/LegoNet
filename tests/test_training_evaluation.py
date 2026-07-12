"""Tests for state-safe in-training evaluation helpers."""

import importlib
import sys
import types
import unittest
from unittest import mock

import config


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

    def test_single_evaluation_restores_dataset_and_rejects_missing_error(self):
        """Single-IoU evaluation also restores the model's training dataset."""
        training_dataset = object()
        model = mock.Mock(dataset=training_dataset)
        self.evaluation.perObject_eval.eval.return_value = []

        relative_error = self.evaluation.evaluate_combined_once(
            object(),
            "loader",
            "sampler",
            model,
            args=object(),
        )

        self.assertIsNone(relative_error)
        self.assertIs(model.dataset, training_dataset)

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
