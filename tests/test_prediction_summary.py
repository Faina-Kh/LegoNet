"""Tests for per-image prediction summary bookkeeping."""

import unittest

import torch

from legonet.eval.evaluation_state import initiate_global_dicts
from legonet.eval.prediction_summary import record_per_image_predictions


class PredictionSummaryTests(unittest.TestCase):
    def test_records_counting_predictions_and_image_average(self):
        state = initiate_global_dicts(initiate=True)

        record_per_image_predictions(
            state,
            "image.jpg",
            [torch.tensor([1.2, 2.8])],
            attributes=False,
        )

        self.assertEqual(state["predicted_counts_any_crop"], [1.0, 3.0])
        self.assertEqual(state["per_im_pred_dict"]["image.jpg"], 2.0)

    def test_records_attribute_predictions_and_aggregates(self):
        state = initiate_global_dicts(initiate=True)

        record_per_image_predictions(
            state,
            "image.jpg",
            [torch.tensor([[0.8, 4.0, 0.4], [0.2, 6.0, 0.6]])],
            attributes=True,
        )

        self.assertEqual(state["predicted_color_any_crop"], [1, 0])
        self.assertEqual(state["TRL_per_im_pred_dict"]["image.jpg"], 10.0)
        self.assertAlmostEqual(state["dia_per_im_pred_dict"]["image.jpg"], 0.5)

    def test_none_outputs_leave_state_unchanged(self):
        state = initiate_global_dicts(initiate=True)
        before = list(state["predicted_counts_any_crop"])

        record_per_image_predictions(
            state, "image.jpg", None, attributes=False
        )

        self.assertEqual(state["predicted_counts_any_crop"], before)
        self.assertNotIn("image.jpg", state["per_im_pred_dict"])


if __name__ == "__main__":
    unittest.main()
