"""Characterization tests for inference dispatch."""

import builtins
import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy  # Keep NumPy loaded outside the temporary dependency-stub context.

import config


def _module(name, **attributes):
    """Create a module populated with lightweight test attributes."""
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


class InferenceTests(unittest.TestCase):
    """Capture the current inference branches without models or datasets."""

    @classmethod
    def setUpClass(cls):
        """Import inference with heavyweight dependencies replaced."""
        torch = _module("torch", no_grad=mock.Mock(), tensor=mock.Mock())
        evaluation_package = _module(
            "legonet.eval",
            attribute_estimation_eval=mock.Mock(),
            detection_eval=mock.Mock(),
            perObject_eval=mock.Mock(),
        )
        dataloader_module = _module(
            "legonet.my_dataloader",
            UnNormalizer=mock.Mock(),
        )
        utils_module = _module("legonet.utils", printf=mock.Mock())
        import legonet

        sys.modules.pop("legonet.inference", None)
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "legonet.eval": evaluation_package,
                    "legonet.my_dataloader": dataloader_module,
                    "legonet.utils": utils_module,
                },
            ),
            mock.patch.object(legonet, "eval", evaluation_package, create=True),
            mock.patch.object(legonet, "utils", utils_module, create=True),
        ):
            cls.inference = importlib.import_module("legonet.inference")

    def setUp(self):
        """Reset evaluator mocks and drawing configuration."""
        for evaluator in (
            self.inference.detection_eval,
            self.inference.both_eval,
            self.inference.attribute_estimation_eval,
        ):
            evaluator.reset_mock()
        self.inference.detection_eval.evaluateMAP_simple.side_effect = None
        self.inference.detection_eval.evaluate_detection_params.side_effect = None
        self.inference.both_eval.eval.side_effect = None
        self.inference.attribute_estimation_eval.eval.side_effect = None
        self.inference.utils.printf.reset_mock()
        config.General.to_draw = False

    def test_detection_inference_records_metrics(self):
        """Detection inference evaluates, prints, and appends its metrics."""
        config.General.NETWORK_TYPE = config.NetworkType.detection
        args = SimpleNamespace(
            evaluate_detection=True,
            have_GT=True,
            eval_detection_params=False,
            evaluate_both=False,
            txt_results="results.txt",
        )
        model = mock.Mock()
        self.inference.detection_eval.evaluateMAP_simple.return_value = (
            0.6,
            0.7,
            0.8,
        )
        mocked_open = mock.mock_open()

        with mock.patch.object(builtins, "open", mocked_open):
            self.inference.run_inference(args, "dataset", "loader", "sampler", model)

        model.eval.assert_called_once_with()
        self.inference.detection_eval.evaluateMAP_simple.assert_called_once()
        mocked_open().write.assert_called_once_with(
            "mAP = 0.600, precision = 0.700, recall = 0.800\n"
        )

    def test_detection_parameter_sweep_uses_configured_lists(self):
        """Parameter-sweep inference forwards all configured thresholds."""
        config.General.NETWORK_TYPE = config.NetworkType.detection
        args = SimpleNamespace(
            evaluate_detection=True,
            have_GT=True,
            eval_detection_params=True,
            evaluate_both=False,
            test_dir="results",
        )
        model = mock.Mock()
        self.inference.detection_eval.evaluate_detection_params.return_value = []

        self.inference.run_inference(args, "dataset", "loader", "sampler", model)

        call = self.inference.detection_eval.evaluate_detection_params.call_args
        self.assertIs(call[1]["iou_threshold"], config.Detection.iou_threshold_list)
        self.assertIs(call[1]["score_threshold"], config.Detection.min_score_list)
        self.assertEqual(call[1]["save_path"], "results")

    def test_combined_inference_dispatches_attribute_evaluation(self):
        """Combined inference invokes per-object evaluation when requested."""
        config.General.NETWORK_TYPE = config.NetworkType.detection_and_counting
        args = SimpleNamespace(
            evaluate_detection=False,
            evaluate_both=True,
        )
        model = mock.Mock()
        self.inference.both_eval.eval.return_value = (0.25,)

        self.inference.run_inference(args, "dataset", "loader", "sampler", model)

        self.inference.both_eval.eval.assert_called_once()
        self.inference.utils.printf.assert_called_once_with(
            "rel error: %.3f \n",
            0.25,
        )

    def test_counting_inference_dispatches_image_evaluation(self):
        """Counting inference uses the per-image attribute evaluator."""
        config.General.NETWORK_TYPE = config.NetworkType.counting_reg
        args = SimpleNamespace()
        model = mock.Mock()
        self.inference.attribute_estimation_eval.eval.return_value = 0.4

        self.inference.run_inference(args, "dataset", "loader", "sampler", model)

        self.inference.attribute_estimation_eval.eval.assert_called_once_with(
            "loader",
            "dataset",
            model,
            args,
        )


if __name__ == "__main__":
    unittest.main()
