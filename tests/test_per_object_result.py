"""Contract tests for legacy per-object evaluation results."""

import unittest

from legonet.eval.per_object_result import (
    AttributeEvaluationResult,
    ClassificationMetrics,
    ClassificationType,
    ClassMetrics,
    PerObjectEvaluationResult,
    PerObjectResultKind,
    RegressionMetrics,
)


class PerObjectEvaluationResultTests(unittest.TestCase):
    """Document both positional layouts without invoking the evaluator."""

    def test_counting_layout_names_and_preserves_all_nine_metrics(self) -> None:
        """The counting contract maps every historical tuple position."""
        legacy = (0.2, 10, 7, 0.7, 0.8, 1.5, 0.4, 2.5, 0.65)

        result = PerObjectEvaluationResult.from_legacy_sequence(legacy)

        self.assertIs(result.kind, PerObjectResultKind.COUNTING)
        self.assertEqual(result.relative_error, 0.2)
        self.assertEqual(result.ground_truth_objects, 10)
        self.assertEqual(result.matched_objects, 7)
        self.assertEqual(result.recall, 0.7)
        self.assertEqual(result.precision, 0.8)
        self.assertEqual(result.mean_absolute_error, 1.5)
        self.assertEqual(result.exact_count_agreement, 0.4)
        self.assertEqual(result.mean_squared_error, 2.5)
        self.assertEqual(result.one_minus_fvu, 0.65)
        self.assertEqual(result.to_legacy_tuple(), legacy)

    def test_attribute_layout_contains_only_trl_relative_error(self) -> None:
        """The attribute contract preserves its single historical value."""
        result = PerObjectEvaluationResult.from_legacy_sequence((0.15,))

        self.assertIs(result.kind, PerObjectResultKind.ATTRIBUTES)
        self.assertEqual(result.relative_error, 0.15)
        self.assertIsNone(result.matched_objects)
        self.assertEqual(
            result.attributes.regression["trl"].mean_relative_error,
            0.15,
        )
        self.assertEqual(result.to_legacy_tuple(), (0.15,))

    def test_legacy_unavailable_sentinels_are_not_reinterpreted(self) -> None:
        """Contract conversion preserves the evaluator's existing sentinels."""
        legacy = (-1, 0, 0, 0.0, 0.0, -1, -1, -1, 2.0)

        result = PerObjectEvaluationResult.from_legacy_sequence(legacy)

        self.assertEqual(result.to_legacy_tuple(), legacy)

    def test_undocumented_layout_is_rejected(self) -> None:
        """Empty and partial outputs cannot silently acquire new semantics."""
        for legacy in ((), (0.2, 10), tuple(range(10))):
            with self.subTest(length=len(legacy)):
                with self.assertRaises(ValueError):
                    PerObjectEvaluationResult.from_legacy_sequence(legacy)

    def test_attribute_metrics_support_named_regression_tasks(self) -> None:
        """Dataset-specific names do not leak into the metric definitions."""
        metrics = AttributeEvaluationResult(
            regression={
                "root_length": RegressionMetrics(
                    eligible_samples=12,
                    evaluated_samples=9,
                    mean_relative_error=0.2,
                ),
                "diameter": RegressionMetrics(mean_squared_error=1.5),
            }
        )

        self.assertEqual(metrics.regression["root_length"].coverage, 0.75)
        self.assertEqual(metrics.regression["diameter"].mean_squared_error, 1.5)

    def test_classification_metrics_support_multiclass_attributes(self) -> None:
        """Class order and per-class metrics work beyond binary labels."""
        metrics = ClassificationMetrics(
            classification_type=ClassificationType.NOMINAL,
            class_names=("healthy", "mild", "severe"),
            confusion_matrix=((4, 1, 0), (1, 3, 1), (0, 1, 4)),
            eligible_samples=18,
            evaluated_samples=15,
            correct_predictions=11,
            accuracy=11 / 15,
            per_class={
                "healthy": ClassMetrics(5, 0.8, 0.8, 0.8),
                "mild": ClassMetrics(5, 0.6, 0.6, 0.6),
                "severe": ClassMetrics(5, 0.8, 0.8, 0.8),
            },
        )

        self.assertAlmostEqual(metrics.coverage, 15 / 18)
        self.assertEqual(metrics.confusion_matrix[1][2], 1)
        self.assertIsNone(metrics.one_minus_fvu)

    def test_binary_classification_can_retain_one_minus_fvu(self) -> None:
        """Binary attributes may expose the legacy variance-based score."""
        metrics = ClassificationMetrics(
            classification_type=ClassificationType.BINARY,
            class_names=("light", "dark"),
            confusion_matrix=((7, 1), (2, 6)),
            one_minus_fvu=0.4,
        )

        self.assertEqual(metrics.one_minus_fvu, 0.4)

    def test_confusion_matrix_requires_documented_class_order(self) -> None:
        """Malformed matrices cannot be reported with misleading labels."""
        with self.assertRaises(ValueError):
            ClassificationMetrics(
                classification_type=ClassificationType.NOMINAL,
                class_names=("a", "b", "c"),
                confusion_matrix=((1, 0), (0, 1)),
            )


if __name__ == "__main__":
    unittest.main()
