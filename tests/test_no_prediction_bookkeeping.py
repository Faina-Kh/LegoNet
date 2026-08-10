"""Tests for no-detection evaluation bookkeeping."""

import unittest

from legonet.eval.evaluation_state import initiate_global_dicts
from legonet.eval.no_prediction_bookkeeping import record_no_predictions


class NoPredictionBookkeepingTests(unittest.TestCase):
    def test_records_counting_sentinels(self):
        state = initiate_global_dicts(initiate=True)

        record_no_predictions(
            state,
            "image.jpg",
            image_gt_average=3,
            image_trl_sum=0,
            image_diameter_average=0,
            attributes=False,
            network_type="per_object_counting",
        )

        self.assertEqual(state["no_predictions"]["image.jpg"]["pred"], [0])
        self.assertEqual(state["no_predictions"]["image.jpg"]["gt_count"], [3])
        self.assertEqual(state["no_predictions"]["image.jpg"]["score"], [-1])

    def test_records_attribute_sentinels_and_color_gt(self):
        state = initiate_global_dicts(initiate=True)

        record_no_predictions(
            state,
            "image.jpg",
            image_gt_average=0.75,
            image_trl_sum=8,
            image_diameter_average=0.4,
            attributes=True,
            network_type="per_object_attributes",
        )

        record = state["no_predictions"]["image.jpg"]
        self.assertEqual(record["TRL_gt"], [8])
        self.assertEqual(record["dia_gt"], [0.4])
        self.assertEqual(record["color_gt"], [0.75])
        self.assertNotIn("pred", record)

    def test_empty_attribute_image_uses_missing_color_sentinel(self):
        state = initiate_global_dicts(initiate=True)

        record_no_predictions(
            state,
            "empty.jpg",
            image_gt_average=0,
            image_trl_sum=0,
            image_diameter_average=0,
            attributes=True,
            network_type="per_object_attributes_multibranch",
        )

        self.assertEqual(
            state["no_predictions"]["empty.jpg"]["color_gt"], [-1]
        )


if __name__ == "__main__":
    unittest.main()
