"""Tests for the canonical per-image attribute evaluation boundary."""

import unittest
from types import SimpleNamespace
from unittest import mock

from legonet.eval import per_image_attribute_eval


class PerImageAttributeEvaluationTests(unittest.TestCase):
    def test_continuous_attribute_returns_named_lower_is_better_metric(self) -> None:
        model = SimpleNamespace(estimator=SimpleNamespace(binary_model=False))
        args = SimpleNamespace(network_type="per_image_estimation_regression")

        with mock.patch.object(
            per_image_attribute_eval._legacy_evaluator,
            "eval",
            return_value=0.25,
        ):
            result = per_image_attribute_eval.evaluate_checkpoint_metrics(
                "loader", "dataset", model, args
            )

        self.assertEqual(result.metric_name, "relative_error")
        self.assertEqual(result.metric_value, 0.25)

    def test_binary_attribute_is_reported_as_classification_error(self) -> None:
        model = SimpleNamespace(estimator=SimpleNamespace(binary_model=True))
        args = SimpleNamespace(network_type="per_image_estimation_keypoints")

        with mock.patch.object(
            per_image_attribute_eval._legacy_evaluator,
            "eval",
            return_value=0.1,
        ):
            result = per_image_attribute_eval.evaluate_checkpoint_metrics(
                "loader", "dataset", model, args
            )

        self.assertEqual(result.metric_name, "classification_error_rate")
        self.assertEqual(result.metric_value, 0.1)


if __name__ == "__main__":
    unittest.main()
