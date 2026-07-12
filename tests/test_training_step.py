"""Branch characterization tests for training loss composition."""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import config


class FakeTensor:
    """Small scalar tensor stand-in supporting loss arithmetic."""

    def __init__(self, value):
        self.value = float(value)
        self.grad_fn = object()

    def mean(self):
        return self

    def item(self):
        return self.value

    def __add__(self, other):
        return FakeTensor(self.value + other.value)

    def __mul__(self, other):
        return FakeTensor(self.value * float(other))

    def __rmul__(self, other):
        return self * other


class TrainingStepTests(unittest.TestCase):
    """Lock down total and component losses for each network family."""

    @classmethod
    def setUpClass(cls):
        torch_module = types.ModuleType("torch")
        torch_module.cuda = SimpleNamespace(is_available=lambda: True)
        torch_module.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=mock.Mock()))
        torch_module.tensor = mock.Mock()
        sys.modules.pop("legonet.training_step", None)
        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            cls.step = importlib.import_module("legonet.training_step")

    def args(self, network_type, **overrides):
        values = {
            "network_type": network_type,
            "loss_weight": 2,
            "color_loss_weight": 100,
            "dia_loss_weight": 10,
            "maps_loss_weight": 1,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_detection_loss(self):
        result = self.step.combine_losses(
            {"classification": FakeTensor(2), "regression": FakeTensor(3)},
            self.args("bbox_detection"),
        )
        self.assertEqual(result.total.item(), 5)
        self.assertEqual(result.values["classification"], 2)

    def test_counting_keypoint_loss(self):
        result = self.step.combine_losses(
            {"l1_estimation": FakeTensor(2), "maps": FakeTensor(1)},
            self.args("per_image_estimation_keypoints"),
        )
        self.assertEqual(result.total.item(), 5)

    def test_per_image_regression_loss(self):
        result = self.step.combine_losses(
            {"reg_estimation": FakeTensor(3)},
            self.args("per_image_estimation_regression"),
        )
        self.assertEqual(result.total.item(), 6)

    def test_combined_keypoint_loss(self):
        config.AttributeEstimation.estimate_type = "withKeyPoints"
        result = self.step.combine_losses(
            {
                "classification": FakeTensor(2),
                "regression": FakeTensor(3),
                "l1_counting": FakeTensor(4),
                "maps": FakeTensor(1),
            },
            self.args("both"),
        )
        self.assertEqual(result.total.item(), 10)

    def test_combined_regression_loss(self):
        config.AttributeEstimation.estimate_type = "reg_fpn_p3_p7_min_sig"
        result = self.step.combine_losses(
            {
                "classification": FakeTensor(2),
                "regression": FakeTensor(3),
                "counting": FakeTensor(4),
            },
            self.args("both"),
        )
        self.assertEqual(result.total.item(), 9)

    def test_root_attribute_keypoint_loss_weights(self):
        config.AttributeEstimation.estimate_type = "withKeyPoints"
        result = self.step.combine_losses(
            {
                "classification": FakeTensor(2),
                "regression": FakeTensor(3),
                "color": FakeTensor(2),
                "maps": FakeTensor(3),
                "length": FakeTensor(4),
                "diameter": FakeTensor(5),
            },
            self.args("both_for_roots_2"),
        )
        self.assertEqual(result.total.item(), 262)

    def test_root_attribute_regression_loss_weights(self):
        config.AttributeEstimation.estimate_type = "reg_fpn_p3_p7_min_sig"
        result = self.step.combine_losses(
            {
                "classification": FakeTensor(2),
                "regression": FakeTensor(3),
                "color": FakeTensor(2),
                "length": FakeTensor(4),
                "diameter": FakeTensor(5),
            },
            self.args("both_for_roots_2"),
        )
        self.assertEqual(result.total.item(), 259)


if __name__ == "__main__":
    unittest.main()
