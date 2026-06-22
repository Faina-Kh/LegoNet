"""Characterization tests for the current runner module.

The runner is imported with lightweight dependency stubs so these tests can run
without PyTorch, model weights, datasets, or a GPU.  The assertions intentionally
describe the current behavior that the runner refactor must preserve.
"""

import importlib
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


class FakeBatchNorm2d:
    """Minimal BatchNorm stand-in used to characterize ``freeze_bn``."""

    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        """Record that the layer was switched to evaluation mode."""
        self.eval_calls += 1


class StopAfterModelBuild(Exception):
    """Stop ``_run`` after data setup, before model execution begins."""


def _module(name, **attributes):
    """Return a module populated with the supplied attributes."""
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _runner_dependency_stubs():
    """Build dependency stubs sufficient to import ``legonet.runner``."""
    torch = _module("torch", manual_seed=mock.Mock())
    torch_nn = _module("torch.nn", BatchNorm2d=FakeBatchNorm2d)
    torch_optim = _module("torch.optim")
    torch_utils = _module("torch.utils")
    torch_utils_data = _module("torch.utils.data", DataLoader=mock.Mock())
    torch.nn = torch_nn
    torch.optim = torch_optim
    torch.utils = torch_utils
    torch_utils.data = torch_utils_data

    transforms = _module("torchvision.transforms", Compose=mock.Mock(side_effect=lambda items: items))
    torchvision = _module("torchvision", transforms=transforms)

    dataloader_names = (
        "KCSVDataset",
        "CocoDataset",
        "collater",
        "Resizer",
        "AspectRatioBasedSampler",
        "Augmenter",
        "Normalizer",
        "UnNormalizer",
        "csv_LCCDataset",
        "LCC_collater",
        "kcsv_collater",
    )
    dataloader_attributes = {name: mock.Mock(name=name) for name in dataloader_names}
    dataloader_module = _module("legonet.my_dataloader", **dataloader_attributes)

    evaluation_module = _module(
        "legonet.eval",
        attribute_estimation_eval=mock.Mock(),
        detection_eval=mock.Mock(),
        perObject_eval=mock.Mock(),
    )
    build_module = _module("legonet.legoNet_build", model_build=mock.Mock())
    weights_module = _module(
        "manage_weights",
        list_checkpoint_modules=mock.Mock(),
        load_submodule_weights=mock.Mock(),
        save_partial_weights=mock.Mock(),
        print_module_names=mock.Mock(),
    )
    utils_module = _module("legonet.utils", printf=mock.Mock())

    return {
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.optim": torch_optim,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torchvision": torchvision,
        "torchvision.transforms": transforms,
        "legonet.eval": evaluation_module,
        "legonet.my_dataloader": dataloader_module,
        "legonet.legoNet_build": build_module,
        "legonet.utils": utils_module,
        "manage_weights": weights_module,
    }


class RunnerCharacterizationTests(unittest.TestCase):
    """Capture inexpensive, externally visible behavior of ``runner.py``."""

    @classmethod
    def setUpClass(cls):
        """Import the runner once with heavyweight dependencies replaced."""
        sys.modules.pop("legonet.runner", None)
        sys.modules.pop("legonet.data_setup", None)
        with mock.patch.dict(sys.modules, _runner_dependency_stubs()):
            cls.runner = importlib.import_module("legonet.runner")
            cls.data_setup = importlib.import_module("legonet.data_setup")

    def setUp(self):
        """Reset dependency mocks before each characterization test."""
        dependency_names = (
            "KCSVDataset",
            "CocoDataset",
            "collater",
            "Resizer",
            "AspectRatioBasedSampler",
            "Augmenter",
            "Normalizer",
            "csv_LCCDataset",
            "LCC_collater",
            "kcsv_collater",
            "DataLoader",
        )
        for name in dependency_names:
            dependency = getattr(self.data_setup, name)
            dependency.reset_mock()
            dependency.side_effect = None
        self.runner.model_build.reset_mock()
        self.runner.model_build.side_effect = None

    def test_tee_writes_and_flushes_every_stream(self):
        """The logging tee duplicates output to all configured streams."""
        first = mock.Mock()
        second = mock.Mock()

        tee = self.runner.Tee(first, second)
        tee.write("training output")
        tee.flush()

        first.write.assert_called_once_with("training output")
        second.write.assert_called_once_with("training output")
        self.assertEqual(first.flush.call_count, 2)
        self.assertEqual(second.flush.call_count, 2)

    def test_unwrap_model_returns_module_when_present(self):
        """Wrapped models expose their underlying model through ``module``."""
        inner_model = object()
        wrapped_model = SimpleNamespace(module=inner_model)

        self.assertIs(self.runner.unwrap_model(wrapped_model), inner_model)

    def test_freeze_bn_only_switches_batch_norm_layers(self):
        """Only BatchNorm2d layers are switched to evaluation mode."""
        batch_norm = FakeBatchNorm2d()
        other_layer = mock.Mock()
        model = mock.Mock()
        model.modules.return_value = [batch_norm, other_layer]

        self.runner.freeze_bn(model)

        self.assertEqual(batch_norm.eval_calls, 1)
        other_layer.eval.assert_not_called()

    def test_kcsv_inference_builds_only_validation_loader(self):
        """KCSV inference uses its validation file and KCSV collater."""
        args = SimpleNamespace(
            dataset_type="kcsv",
            network_type="both",
            run_script="Inference",
            val_file="validation.txt",
            kcsv_classes="classes.txt",
            pre_process="torch_like",
            num_workers=0,
        )
        validation_dataset = object()
        validation_sampler = object()
        self.data_setup.KCSVDataset.return_value = validation_dataset
        self.data_setup.AspectRatioBasedSampler.return_value = validation_sampler
        self.runner.model_build.side_effect = StopAfterModelBuild

        with self.assertRaises(StopAfterModelBuild):
            self.runner._run(args)

        self.data_setup.KCSVDataset.assert_called_once()
        dataset_call = self.data_setup.KCSVDataset.call_args
        self.assertEqual(dataset_call[1]["input_file"], "validation.txt")
        self.assertEqual(dataset_call[1]["class_list"], "classes.txt")
        self.data_setup.AspectRatioBasedSampler.assert_called_once_with(
            validation_dataset,
            batch_size=1,
            drop_last=False,
            do_shuffle=False,
        )
        self.data_setup.DataLoader.assert_called_once_with(
            validation_dataset,
            num_workers=0,
            collate_fn=self.data_setup.kcsv_collater,
            batch_sampler=validation_sampler,
        )
        self.runner.model_build.assert_called_once_with(args, None, validation_dataset)

    def test_csv_lcc_inference_builds_only_validation_loader(self):
        """Roots CSV inference uses its validation files and LCC collater."""
        args = SimpleNamespace(
            dataset_type="csv_LCC",
            run_script="Inference",
            val_csv_leaf_number_file="validation.csv",
            val_csv_leaf_location_file="validation_points.csv",
            val_json_file="validation.json",
            pre_process="torch_like",
            base_dir="dataset",
            have_GT=True,
            num_workers=0,
        )
        validation_dataset = object()
        validation_sampler = object()
        self.data_setup.csv_LCCDataset.return_value = validation_dataset
        self.data_setup.AspectRatioBasedSampler.return_value = validation_sampler
        self.runner.model_build.side_effect = StopAfterModelBuild

        with self.assertRaises(StopAfterModelBuild):
            self.runner._run(args)

        self.data_setup.csv_LCCDataset.assert_called_once()
        dataset_call = self.data_setup.csv_LCCDataset.call_args
        self.assertEqual(dataset_call[0][:2], ("validation.csv", "validation_points.csv"))
        self.assertEqual(dataset_call[1]["pre_process"], "keras_like")
        self.assertEqual(dataset_call[1]["ann_type"], "count")
        self.assertEqual(dataset_call[1]["json_file"], "validation.json")
        self.assertEqual(dataset_call[1]["base_dir"], "dataset")
        self.assertTrue(dataset_call[1]["have_GT"])
        self.data_setup.AspectRatioBasedSampler.assert_called_once_with(
            validation_dataset,
            batch_size=1,
            drop_last=False,
            do_shuffle=False,
        )
        self.data_setup.DataLoader.assert_called_once_with(
            validation_dataset,
            num_workers=0,
            collate_fn=self.data_setup.LCC_collater,
            batch_sampler=validation_sampler,
        )
        self.runner.model_build.assert_called_once_with(args, None, validation_dataset)

    def test_roots_json_inference_builds_kcsv_validation_loader(self):
        """Roots JSON inference retains the KCSV dataset implementation."""
        args = SimpleNamespace(
            dataset_type="roots_json",
            network_type="bbox_detection",
            run_script="Inference",
            val_json_file="validation.json",
            pre_process="torch_like",
            base_dir="dataset",
            have_GT=True,
            num_workers=0,
        )
        validation_dataset = object()
        validation_sampler = object()
        self.data_setup.KCSVDataset.return_value = validation_dataset
        self.data_setup.AspectRatioBasedSampler.return_value = validation_sampler
        self.runner.model_build.side_effect = StopAfterModelBuild

        with self.assertRaises(StopAfterModelBuild):
            self.runner._run(args)

        self.data_setup.KCSVDataset.assert_called_once()
        dataset_call = self.data_setup.KCSVDataset.call_args
        self.assertEqual(dataset_call[1]["input_file"], "validation.json")
        self.assertEqual(dataset_call[1]["dataset_type"], "roots_json")
        self.assertEqual(dataset_call[1]["base_dir"], "dataset")
        self.assertTrue(dataset_call[1]["have_GT"])
        self.data_setup.DataLoader.assert_called_once_with(
            validation_dataset,
            num_workers=0,
            collate_fn=self.data_setup.kcsv_collater,
            batch_sampler=validation_sampler,
        )
        self.runner.model_build.assert_called_once_with(args, None, validation_dataset)

    def test_coco_training_builds_train_and_validation_loaders(self):
        """COCO training creates both loaders with the detection collater."""
        args = SimpleNamespace(
            dataset_type="coco",
            dataset_path="dataset",
            run_script="Training",
            batch_size=2,
            num_workers=3,
        )
        train_dataset = object()
        validation_dataset = object()
        train_sampler = object()
        validation_sampler = object()
        self.data_setup.CocoDataset.side_effect = [train_dataset, validation_dataset]
        self.data_setup.AspectRatioBasedSampler.side_effect = [train_sampler, validation_sampler]
        self.runner.model_build.side_effect = StopAfterModelBuild

        with self.assertRaises(StopAfterModelBuild):
            self.runner._run(args)

        self.assertEqual(self.data_setup.CocoDataset.call_count, 2)
        self.assertEqual(self.data_setup.CocoDataset.call_args_list[0][1]["set_name"], "train")
        self.assertEqual(self.data_setup.CocoDataset.call_args_list[1][1]["set_name"], "val")
        self.assertIs(args.collater, self.data_setup.collater)
        self.assertEqual(self.data_setup.DataLoader.call_count, 2)
        self.data_setup.DataLoader.assert_has_calls(
            [
                mock.call(
                    train_dataset,
                    num_workers=3,
                    collate_fn=self.data_setup.collater,
                    batch_sampler=train_sampler,
                ),
                mock.call(
                    validation_dataset,
                    num_workers=3,
                    collate_fn=self.data_setup.collater,
                    batch_sampler=validation_sampler,
                ),
            ]
        )
        self.runner.model_build.assert_called_once_with(args, train_dataset, validation_dataset)


if __name__ == "__main__":
    unittest.main()
