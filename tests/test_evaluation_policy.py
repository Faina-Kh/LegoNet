"""Tests for evaluator image-inclusion policies."""

import unittest

from legonet.eval.evaluation_policy import EvaluationTask, should_include_image


class EvaluationPolicyTests(unittest.TestCase):
    """Verify task-specific handling of images without usable targets."""

    def test_detection_includes_all_images(self) -> None:
        """Empty images remain part of bounding-box detection metrics."""
        self.assertTrue(
            should_include_image(EvaluationTask.DETECTION, [], [])
        )
        self.assertTrue(
            should_include_image(EvaluationTask.DETECTION, [object()], [])
        )

    def test_per_object_requires_annotated_object(self) -> None:
        """Per-object metrics require a box with per-object annotations."""
        self.assertFalse(
            should_include_image(EvaluationTask.PER_OBJECT, [], [])
        )
        self.assertFalse(
            should_include_image(
                EvaluationTask.PER_OBJECT,
                [object()],
                [],
            )
        )
        self.assertTrue(
            should_include_image(
                EvaluationTask.PER_OBJECT,
                [object()],
                [object()],
            )
        )


if __name__ == "__main__":
    unittest.main()
