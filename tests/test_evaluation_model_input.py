"""Tests for per-object evaluation model inputs."""

import unittest

import torch

from legonet.eval.model_input import build_per_object_model_input


class EvaluationModelInputTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.ones((1, 2, 2), dtype=torch.float64)
        self.boxes = torch.tensor([[0, 0, 1, 1]])

    def test_inference_uses_no_annotations(self):
        model_input = build_per_object_model_input(
            self.image, {}, [3], have_gt=False, device="cpu"
        )

        self.assertEqual(model_input[0].dtype, torch.float32)
        self.assertIsNone(model_input[1])
        torch.testing.assert_close(model_input[2], torch.tensor([3]))

    def test_gt_evaluation_includes_bbox_and_points(self):
        points = [{"x": 1, "y": 1}]
        model_input = build_per_object_model_input(
            self.image,
            {"bbox_annot": self.boxes, "points_annot": points},
            [0],
            have_gt=True,
            device="cpu",
        )

        self.assertIs(model_input[1][0], self.boxes)
        self.assertIs(model_input[1][1], points)

    def test_gt_evaluation_without_points_uses_none(self):
        model_input = build_per_object_model_input(
            self.image,
            {"bbox_annot": self.boxes},
            [0],
            have_gt=True,
            device="cpu",
        )

        self.assertIs(model_input[1][0], self.boxes)
        self.assertIsNone(model_input[1][1])


if __name__ == "__main__":
    unittest.main()
