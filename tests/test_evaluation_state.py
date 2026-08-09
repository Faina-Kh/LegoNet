"""Tests for per-object evaluation-state construction."""

import unittest

from legonet import config
from legonet.eval.evaluation_state import initiate_global_dicts


class EvaluationStateTests(unittest.TestCase):
    """Verify common and task-specific state fields."""

    def setUp(self) -> None:
        self.original_type = config.Detect_and_Estimate.type

    def tearDown(self) -> None:
        config.Detect_and_Estimate.type = self.original_type

    def test_initializes_common_state(self) -> None:
        config.Detect_and_Estimate.type = "bbox_detection"

        state = initiate_global_dicts(initiate=True)

        self.assertEqual(state["found_orig_objects"], 0)
        self.assertEqual(state["detections_data_any_crop"], {})
        self.assertNotIn("all_predicted_TRL", state)

    def test_adds_counting_fields_for_one_image(self) -> None:
        config.Detect_and_Estimate.type = "per_object_counting"
        state = initiate_global_dicts(initiate=True)

        result = initiate_global_dicts(state, "grapes.jpg")

        self.assertIs(result, state)
        self.assertEqual(
            state["detections_data_any_crop"]["grapes.jpg"]["pred"], []
        )
        self.assertEqual(state["not_found_gt"]["grapes.jpg"]["pred"], [])

    def test_adds_independent_attribute_records_per_image(self) -> None:
        config.Detect_and_Estimate.type = "per_object_attributes_multibranch"
        state = initiate_global_dicts(initiate=True)

        initiate_global_dicts(state, "first.jpg")
        initiate_global_dicts(state, "second.jpg")
        state["detections_data_any_crop"]["first.jpg"]["color_pred"].append(1)

        self.assertIn("TRL_per_im_gt_sum_dict", state)
        self.assertEqual(
            state["detections_data_any_crop"]["second.jpg"]["color_pred"],
            [],
        )
        self.assertEqual(
            state["not_found_gt"]["first.jpg"]["color_pred"], []
        )

    def test_requires_state_and_image_name_when_extending(self) -> None:
        with self.assertRaises(ValueError):
            initiate_global_dicts()

        state = initiate_global_dicts(initiate=True)
        with self.assertRaises(ValueError):
            initiate_global_dicts(state)


if __name__ == "__main__":
    unittest.main()
