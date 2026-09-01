"""Regression tests for empty root samples during per-object training."""

import unittest
from unittest import mock

import torch

from legonet import config
from legonet.models import model_per_object_attributes
from legonet.models import model_per_object_attributes_multibranch


class _EmptyDetector(torch.nn.Module):
    """Return no predicted boxes while remaining frozen in evaluation mode."""

    def forward(self, inputs):
        del inputs
        return torch.empty(0), torch.empty(0), torch.empty((0, 4))


class EmptyRootTrainingSampleTests(unittest.TestCase):
    """Empty root images must skip losses without returning implicit None."""

    def _build_model(self, model_module):
        model = model_module.PerObjectEstimate.__new__(
            model_module.PerObjectEstimate
        )
        torch.nn.Module.__init__(model)
        model.bbox_detection = _EmptyDetector()
        model.bbox_detection.eval()
        model.training = True
        return model

    def _run_empty_sample(self, model_module):
        model = self._build_model(model_module)
        image = torch.zeros((1, 3, 16, 16))
        annotations = [torch.empty((1, 0, 6)), None]
        with mock.patch.object(config.General, "device", torch.device("cpu")):
            return model([image, annotations, torch.tensor([0])])

    def test_standard_keypoint_model_returns_six_empty_losses(self):
        with mock.patch.object(
            config.AttributeEstimation,
            "estimate_type",
            "withKeyPoints",
        ):
            result = self._run_empty_sample(model_per_object_attributes)

        self.assertEqual(result, (None, None, None, None, None, None))

    def test_standard_regression_model_returns_five_empty_losses(self):
        with mock.patch.object(
            config.AttributeEstimation,
            "estimate_type",
            "reg_fpn_p3_p7_min_sig",
        ):
            result = self._run_empty_sample(model_per_object_attributes)

        self.assertEqual(result, (None, None, None, None, None))

    def test_multibranch_model_returns_six_empty_losses(self):
        result = self._run_empty_sample(
            model_per_object_attributes_multibranch
        )

        self.assertEqual(result, (None, None, None, None, None, None))


if __name__ == "__main__":
    unittest.main()
