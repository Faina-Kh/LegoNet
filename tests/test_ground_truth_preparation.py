"""Tests for per-image GT preparation."""

import unittest
import numpy as np

from legonet.eval.evaluation_state import initiate_global_dicts
from legonet.eval.ground_truth_preparation import prepare_image_ground_truth


class GroundTruthPreparationTests(unittest.TestCase):
    def test_prepares_annotated_boxes_and_attribute_statistics(self):
        state = initiate_global_dicts(initiate=True)
        boxes = np.array([[0, 0, 10, 10, 0, 11], [20, 20, 30, 30, 0, 22]])
        counts = np.array([[1, 0, 11, 4.0, 0.5]])
        result = prepare_image_ground_truth(
            state, "image.jpg", boxes, counts, have_gt=True, attributes=True
        )
        self.assertTrue(result.include_image)
        self.assertEqual(result.all_boxes.shape[0], 2)
        self.assertEqual(result.annotated_boxes.shape[0], 1)
        self.assertEqual(result.image_trl_sum, 4.0)
        self.assertEqual(state["gt_objects_withGTpoints"], 1)

    def test_skips_boxes_without_object_annotations(self):
        state = initiate_global_dicts(initiate=True)
        boxes = np.array([[0, 0, 10, 10, 0, 11]])
        result = prepare_image_ground_truth(
            state, "image.jpg", boxes, [], have_gt=True, attributes=False
        )
        self.assertFalse(result.include_image)
        self.assertEqual(result.skip_reason, "no_annotated_boxes")

    def test_inference_without_gt_remains_eligible(self):
        state = initiate_global_dicts(initiate=True)
        result = prepare_image_ground_truth(
            state, "image.jpg", None, [], have_gt=False, attributes=True
        )
        self.assertTrue(result.include_image)
        self.assertEqual(state["per_im_gt_avg_dict"]["image.jpg"], 0)


if __name__ == "__main__":
    unittest.main()
