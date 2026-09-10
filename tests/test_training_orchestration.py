"""Tests for epoch-level training evaluation orchestration."""

import unittest
import io
from contextlib import redirect_stdout
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

    def test_epoch_summary_identifies_every_value_as_a_mean_loss(self):
        """Component summaries distinguish losses from evaluation metrics."""
        with mock.patch.object(training.utils, "printf") as printf:
            training._print_epoch_summary(
                4,
                {
                    "reg_estimation": [2.0, 4.0],
                    "maps": [1.0, 3.0],
                    "unused": [],
                },
            )

        self.assertEqual(
            printf.call_args.args,
            (
                "Epoch %d summary: %s\n",
                4,
                "reg_estimation mean loss 3.00000, maps mean loss 2.00000",
            ),
        )

    def test_learning_rate_report_uses_post_scheduler_optimizer_values(self):
        """The epoch report displays every optimizer parameter-group rate."""
        optimizer = SimpleNamespace(
            param_groups=[{"lr": 1e-6}, {"lr": 2.5e-7}]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            training._print_learning_rates(12, optimizer)

        self.assertEqual(
            output.getvalue(),
            "Learning rate after epoch 12: group 0: 1e-06, group 1: 2.5e-07\n",
        )

    def test_running_loss_restarts_at_each_epoch(self):
        """Progress output reports the current epoch mean, not a rolling window."""
        args = SimpleNamespace(
            epochs=2,
            network_type="per_image_estimation",
            estimate_type="withKeyPoints",
            evaluate_detection=False,
            choose_epoch_by_IoUavg=False,
        )
        model = mock.Mock()
        sampler = SimpleNamespace(groups=[[0]])
        results = [
            training.LossResult(
                SimpleNamespace(item=lambda value=value: value),
                {"l1_estimation": value, "maps": 0.0},
            )
            for value in (10.0, 20.0)
        ]

        with (
            mock.patch.object(training.optim, "Adam"),
            mock.patch.object(
                training.optim.lr_scheduler, "ReduceLROnPlateau"
            ) as reduce_on_plateau,
            mock.patch.object(training, "run_training_step", side_effect=results),
            mock.patch.object(training, "_print_step") as print_step,
            mock.patch.object(training, "_print_epoch_summary"),
            mock.patch.object(training, "_evaluate_per_image_attribute_epoch"),
            mock.patch.object(training, "_print_best_training_error"),
        ):
            training.train_model(
                args,
                model,
                dataset_train=[object()],
                dataset_val=[object()],
                sampler=sampler,
                sampler_val=object(),
                dataloader_train=[{}],
                dataloader_val=object(),
            )

        self.assertEqual(
            [call.args[4] for call in print_step.call_args_list],
            [10.0, 20.0],
        )
        self.assertEqual(
            reduce_on_plateau.call_args.kwargs,
            {"patience": 3, "verbose": True},
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

    def test_detection_epoch_reports_validation_and_saves_first_checkpoint(self):
        """The first detector epoch is visible and always establishes a best model."""
        args = SimpleNamespace(
            dataset_type="roots_json",
            evaluate_detection=True,
        )
        metrics = SimpleNamespace(
            mean_average_precision=0.0,
            precision=0.0,
            recall=0.0,
        )
        output = io.StringIO()

        with mock.patch.object(
            training, "evaluate_detection", return_value=metrics
        ), mock.patch.object(training, "save_epoch_checkpoint") as save_checkpoint:
            with redirect_stdout(output):
                training._evaluate_detection_epoch(
                    args,
                    epoch=0,
                    model="model",
                    dataset_val=["a", "b"],
                    dataloader_val="loader",
                    sampler_val="sampler",
                    best=training.BestMetrics(),
                )

        report = output.getvalue()
        self.assertIn("Starting bounding-box evaluation", report)
        self.assertIn("2 images", report)
        self.assertIn("New best validation mAP: 0.000", report)
        self.assertIn("Finished validation evaluation", report)
        save_checkpoint.assert_called_once_with(
            "model", 0, replace_existing=True
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
        self.assertIn("0.300", message)
        self.assertIn("previous: 0.400", message)
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
            "Best validation length_relative_error: 0.250, achieved at epoch 17.",
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
