import unittest

import numpy as np

from legonet.models.keypoint_utils import KeypointUtilitiesMixin


class KeypointUtilitiesTests(unittest.TestCase):
    def test_keypoint_map_preserves_input_annotations(self):
        points = [{"x": 16.0, "y": 8.0}]

        target = KeypointUtilitiesMixin().compute_keypoints_targets_multi_maps(
            (32, 64, 3), points, radius=(5, 5), pyramid_level=3
        )

        self.assertEqual(target.shape, (4, 8))
        self.assertEqual(float(target.max()), 1.0)
        self.assertEqual(points, [{"x": 16.0, "y": 8.0}])

    def test_empty_keypoints_preserve_legacy_return_shape(self):
        target = KeypointUtilitiesMixin().compute_keypoints_targets_multi_maps((32, 64, 3), [])

        self.assertIsInstance(target, list)
        np.testing.assert_array_equal(target[0], np.zeros((4, 8)))

    def test_nmcs_suppresses_subset_box(self):
        boxes = np.zeros((2, 4))
        points = [
            {"x": [1.0, 2.0], "y": [1.0, 2.0]},
            {"x": [1.0], "y": [1.0]},
        ]

        self.assertEqual(KeypointUtilitiesMixin.nmcs(boxes, points), [True, False])

    def test_attribute_points_require_matching_bbox_id(self):
        points = [
            {"x": 2.0, "y": 3.0, "bbox_id": 7.0},
            {"x": 2.0, "y": 3.0, "bbox_id": 8.0},
        ]
        boxes = np.array([[0.0, 0.0, 10.0, 10.0, 7.0]])

        assigned = KeypointUtilitiesMixin.find_points_in_bbox(
            None, points, boxes, scale=2.0, network_type="per_object_attributes"
        )

        self.assertEqual(assigned, [{"x": [4.0], "y": [6.0]}])
        self.assertEqual(points[0]["x"], 4.0)
        self.assertEqual(points[0]["y"], 6.0)

    def test_counting_points_use_geometric_containment_only(self):
        points = [
            {"x": 2.0, "y": 3.0, "bbox_id": 7.0},
            {"x": 2.0, "y": 3.0, "bbox_id": 8.0},
        ]
        boxes = np.array([[0.0, 0.0, 10.0, 10.0, 7.0]])

        assigned = KeypointUtilitiesMixin.find_points_in_bbox(
            None, points, boxes, scale=1.0, network_type="per_object_counting"
        )

        self.assertEqual(assigned, [{"x": [2.0, 2.0], "y": [3.0, 3.0]}])


if __name__ == "__main__":
    unittest.main()
