"""Tests for per-image evaluation input preparation."""

import os
import unittest
from types import SimpleNamespace

import torch

from legonet.eval.image_context import prepare_image_context


class EvaluationImageContextTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            image_ids=[4],
            img_info={4: {"name": "sample.jpg"}},
            base_dir="dataset",
        )
        self.sampler = SimpleNamespace(groups=[[0]])
        self.data = {"img": torch.ones((1, 2, 2)), "scale": [2.0]}

    def test_prepares_image_identity_and_ground_truth(self):
        boxes = {4: ["boxes"]}
        counts = {4: ["counts"]}

        context = prepare_image_context(
            self.dataset,
            self.sampler,
            0,
            self.data,
            have_gt=True,
            all_box_annotations=boxes,
            all_count_annotations=counts,
        )

        self.assertEqual(context.image_id, 4)
        self.assertEqual(context.image_name, "sample.jpg")
        self.assertTrue(
            context.image_path.endswith(os.path.join("dataset", "sample.jpg"))
        )
        self.assertEqual(context.box_annotations, "boxes")
        self.assertEqual(context.count_annotations, "counts")
        self.assertIsNot(context.image, self.data["img"])

    def test_prepares_inference_context_without_ground_truth(self):
        context = prepare_image_context(
            self.dataset, self.sampler, 0, self.data, have_gt=False
        )

        self.assertIsNone(context.box_annotations)
        self.assertEqual(context.count_annotations, [])

    def test_requires_annotation_collections_when_ground_truth_is_enabled(self):
        with self.assertRaisesRegex(ValueError, "annotation collections"):
            prepare_image_context(
                self.dataset, self.sampler, 0, self.data, have_gt=True
            )


if __name__ == "__main__":
    unittest.main()
