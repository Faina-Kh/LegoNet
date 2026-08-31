"""Tests for epoch-level training evaluation orchestration."""

import unittest
from types import SimpleNamespace
from unittest import mock

from legonet import training


class TrainingOrchestrationTests(unittest.TestCase):
    """Verify frozen detector evaluation timing and scope."""

    def test_standalone_detector_is_not_in_frozen_detector_networks(self):
        """Bounding-box training must leave the standalone detector trainable."""
        self.assertNotIn(
            "bbox_detection",
            training.PER_OBJECT_NETWORKS_WITH_FROZEN_DETECTOR,
        )
        self.assertIn(
            "per_object_attributes",
            training.PER_OBJECT_NETWORKS_WITH_FROZEN_DETECTOR,
        )

    def test_per_object_detector_is_evaluated_once_before_training(self):
        args = SimpleNamespace(
            network_type="per_object_counting",
            evaluate_detection=True,
        )
        metrics = SimpleNamespace(
            mean_average_precision=0.6,
            precision=0.7,
            recall=0.8,
        )

        with mock.patch.object(
            training, "evaluate_detection", return_value=metrics
        ) as evaluate_detection:
            training._evaluate_frozen_detector_before_training(
                args, "model", "dataset", "loader", "sampler"
            )

        evaluate_detection.assert_called_once_with(
            "dataset", "loader", "sampler", "model"
        )

    def test_detector_evaluation_is_skipped_when_disabled(self):
        args = SimpleNamespace(
            network_type="per_object_attributes",
            evaluate_detection=False,
        )

        with mock.patch.object(training, "evaluate_detection") as evaluate_detection:
            training._evaluate_frozen_detector_before_training(
                args, "model", "dataset", "loader", "sampler"
            )

        evaluate_detection.assert_not_called()

    def test_best_checkpoint_notice_describes_replacement(self):
        """A lower validation error clearly announces checkpoint replacement."""
        with mock.patch("builtins.print") as print_mock:
            training._print_best_error_checkpoint_notice(12, 0.4, 0.3)

        message = print_mock.call_args.args[0]
        self.assertIn("0.300000", message)
        self.assertIn("previous: 0.400000", message)
        self.assertIn("epoch 12", message)
        self.assertIn("Replacing", message)

    def test_best_training_error_reports_selected_epoch(self):
        """Training completion reports the best error and its epoch."""
        args = SimpleNamespace(choose_epoch_by_IoUavg=False)
        best = training.BestMetrics(
            checkpoint_metric_name="length_relative_error",
            checkpoint_metric_value=0.25,
            checkpoint_metric_epoch=17,
        )

        with mock.patch("builtins.print") as print_mock:
            training._print_best_training_error(args, best)

        self.assertEqual(
            print_mock.call_args.args[0],
            "Best validation length_relative_error: 0.250000, achieved at epoch 17.",
        )

    def test_checkpoint_comparison_minimizes_error(self):
        self.assertFalse(
            training._is_better_checkpoint_error(
                0.8,
                0.7,
            )
        )
        self.assertTrue(
            training._is_better_checkpoint_error(
                0.2,
                0.3,
            )
        )

    def test_best_training_error_reports_missing_validation(self):
        """Training completion is explicit when no valid error was produced."""
        args = SimpleNamespace(choose_epoch_by_IoUavg=False)

        with mock.patch("builtins.print") as print_mock:
            training._print_best_training_error(args, training.BestMetrics())

        self.assertIn("without a valid", print_mock.call_args.args[0])

    def test_empty_iou_sweep_is_reported_without_selecting_checkpoint(self):
        """Missing sweep metrics do not crash evaluation or replace weights."""
        args = SimpleNamespace(eval_in_train=True, choose_epoch_by_IoUavg=True)
        sweep = SimpleNamespace(measurements=(), average_relative_error=None)

        with mock.patch.object(training.torch.cuda, "is_available", return_value=True):
            with mock.patch.object(
                training, "evaluate_combined_iou_sweep", return_value=sweep
            ), mock.patch.object(
                training, "save_epoch_checkpoint"
            ) as save_checkpoint, mock.patch("builtins.print") as print_mock:
                training._evaluate_combined_epoch(
                    args,
                    epoch=3,
                    model="model",
                    dataset_val="dataset",
                    dataloader_val="loader",
                    sampler_val="sampler",
                    best=training.BestMetrics(),
                )

        self.assertIn("average_error=n/a", print_mock.call_args.args[0])
        save_checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
