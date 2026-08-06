"""Tests for reusable classification-attribute metrics."""

import unittest

from legonet.eval.classification_metrics import (
    compute_classification_metrics,
    decode_class_predictions,
)
from legonet.eval.per_object_result import ClassificationType


class ClassificationMetricsTests(unittest.TestCase):
    """Verify binary and multiclass metric semantics."""

    def test_binary_accuracy_uses_evaluated_predictions(self) -> None:
        metrics = compute_classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            class_labels=(0, 1),
            class_names=("non_white", "white"),
            classification_type=ClassificationType.BINARY,
            eligible_samples=5,
        )

        self.assertEqual(metrics.correct_predictions, 3)
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertEqual(metrics.error_rate, 0.25)
        self.assertEqual(metrics.coverage, 0.8)
        self.assertEqual(metrics.confusion_matrix, ((1, 1), (0, 2)))
        self.assertAlmostEqual(metrics.balanced_accuracy, 0.75)
        self.assertAlmostEqual(metrics.per_class["white"].precision, 2 / 3)
        self.assertAlmostEqual(metrics.one_minus_fvu, 0.0)

    def test_decodes_one_binary_prediction_per_object(self) -> None:
        predictions = decode_class_predictions(
            [[0.2], [0.5], [0.9]], ClassificationType.BINARY
        )

        self.assertEqual(predictions.tolist(), [0, 1, 1])

    def test_decodes_one_multiclass_prediction_per_object(self) -> None:
        predictions = decode_class_predictions(
            [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
            ClassificationType.NOMINAL,
        )

        self.assertEqual(predictions.tolist(), [0, 2])

    def test_nominal_multiclass_metrics_do_not_compute_fvu(self) -> None:
        metrics = compute_classification_metrics(
            [0, 1, 2, 2],
            [0, 2, 2, 1],
            class_labels=(0, 1, 2),
            class_names=("healthy", "mild", "severe"),
        )

        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.micro_f1, 0.5)
        self.assertAlmostEqual(metrics.macro_recall, 0.5)
        self.assertIsNone(metrics.one_minus_fvu)

    def test_ordinal_multiclass_uses_configured_class_order_for_fvu(self) -> None:
        metrics = compute_classification_metrics(
            ["low", "medium", "high"],
            ["medium", "medium", "high"],
            class_labels=("low", "medium", "high"),
            classification_type=ClassificationType.ORDINAL,
        )

        self.assertAlmostEqual(metrics.one_minus_fvu, 0.5)

    def test_empty_evaluation_has_explicitly_unavailable_scores(self) -> None:
        metrics = compute_classification_metrics(
            [], [], class_labels=(0, 1), eligible_samples=3
        )

        self.assertEqual(metrics.evaluated_samples, 0)
        self.assertEqual(metrics.coverage, 0.0)
        self.assertIsNone(metrics.accuracy)
        self.assertIsNone(metrics.error_rate)
        self.assertIsNone(metrics.macro_f1)

    def test_rejects_unknown_observed_label(self) -> None:
        with self.assertRaises(ValueError):
            compute_classification_metrics([0], [2], class_labels=(0, 1))

    def test_rejects_misaligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_classification_metrics([0, 1], [0], class_labels=(0, 1))


if __name__ == "__main__":
    unittest.main()
