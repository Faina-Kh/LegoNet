"""Characterization tests for the current runner module.

The runner is imported with lightweight dependency stubs so these tests can run
without PyTorch, model weights, datasets, or a GPU.  The assertions intentionally
describe the current behavior that the runner refactor must preserve.
"""

import importlib
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy  # Keep NumPy loaded outside the temporary dependency-stub context.


class FakeBatchNorm2d:
    """Minimal BatchNorm stand-in used to characterize ``freeze_bn``."""

    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        """Record that the layer was switched to evaluation mode."""
        self.eval_calls += 1


class StopAfterModelBuild(Exception):
    """Stop ``_run`` after data setup, before model execution begins."""


class StopAfterModelTo(Exception):
    """Stop ``_run`` after weight setup, before training or inference."""


def _module(name, **attributes):
    """Return a module populated with the supplied attributes."""
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _runner_dependency_stubs():
    """Build dependency stubs sufficient to import ``legonet.runner``."""
    torch = _module("torch", manual_seed=mock.Mock(), load=mock.Mock())
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
        per_image_attribute_eval=mock.Mock(),
        detection_eval=mock.Mock(),
        perObject_eval=mock.Mock(),
    )
    build_module = _module("legonet.legoNet_build", model_build=mock.Mock())
    weights_module = _module(
        "legonet.manage_weights",
        list_checkpoint_modules=mock.Mock(),
        load_submodule_weights=mock.Mock(),
        save_partial_weights=mock.Mock(),
        print_module_names=mock.Mock(),
        validate_checkpoint_modules=mock.Mock(),
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
        "legonet.manage_weights": weights_module,
    }


class RunnerCharacterizationTests(unittest.TestCase):
    """Capture inexpensive, externally visible behavior of ``runner.py``."""

    @classmethod
    def setUpClass(cls):
        """Import the runner once with heavyweight dependencies replaced."""
        sys.modules.pop("legonet.runner", None)
        sys.modules.pop("legonet.data_setup", None)
        sys.modules.pop("legonet.model_setup", None)
        with mock.patch.dict(sys.modules, _runner_dependency_stubs()):
            cls.runner = importlib.import_module("legonet.runner")
            cls.data_setup = importlib.import_module("legonet.data_setup")
            cls.model_setup = importlib.import_module("legonet.model_setup")

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
        self.model_setup.torch.load.reset_mock()
        self.model_setup.torch.load.side_effect = None
        for name in (
            "list_checkpoint_modules",
            "load_submodule_weights",
            "save_partial_weights",
            "print_module_names",
        ):
            dependency = getattr(self.model_setup, name)
            dependency.reset_mock()
            dependency.side_effect = None

    def _weight_args(self, **overrides):
        """Return the common arguments needed to reach weight setup."""
        values = {
            "load_weights": True,
            "load_partial_weights": False,
            "load_full_model_weights": False,
            "save_from_model_file": False,
            "myExpPath": "experiment",
            "network_type": "per_image_estimation",
            "estimate_type": "reg_fpn_p3_p7_min_sig",
            "freeze_detection": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _run_through_weight_setup(self, args, model, stop_after_to=True):
        """Run with data/model stubs until weight setup has completed."""
        data = SimpleNamespace(
            dataset_train=None,
            dataset_val=object(),
            sampler=None,
            sampler_val=object(),
            dataloader_train=None,
            dataloader_val=object(),
        )
        self.runner.model_build.return_value = model
        if stop_after_to:
            model.to.side_effect = StopAfterModelTo

        with mock.patch.object(self.runner, "build_data", return_value=data):
            if stop_after_to:
                with self.assertRaises(StopAfterModelTo):
                    self.runner._run(args)
            else:
                self.runner._run(args)

    def test_run_parameters_are_grouped_and_complete_json_is_saved(self):
        """Console output stays concise while JSON retains resolved settings."""
        args = SimpleNamespace(
            _invocation_argv=["--dataset-name", "grapes"],
            run_script="Inference",
            dataset_name="grapes",
            network_type="per_object_counting",
            estimate_type="withKeyPoints",
            val_set="Test",
            STORAGE_PATH=r"D:\Faina\LegoNet",
            txt_results="",
            weights_mode="full",
            full_model_weights="counting.pt",
            have_GT=True,
            evaluate_detection=True,
            to_draw=False,
            batch_size=1,
            num_workers=0,
            internal_detail={"value": numpy.int64(2)},
        )

        with TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.txt"
            args.txt_results = str(results_path)
            output = io.StringIO()
            with (
                mock.patch.object(
                    self.runner.config.General, "experiment_path", "experiment"
                ),
                mock.patch.object(self.runner.config.General, "device", "cpu"),
                redirect_stdout(output),
            ):
                self.runner.print_args(args, str(results_path))

            summary = results_path.read_text(encoding="utf-8")
            configuration = json.loads(
                (Path(directory) / "run_configuration.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(output.getvalue(), summary)
        self.assertIn("Run\n  Mode: Inference", summary)
        self.assertIn("Estimate type: keypoints", summary)
        self.assertNotIn("withKeyPoints", summary)
        self.assertIn("Storage and output", summary)
        self.assertIn("Full checkpoint: counting.pt", summary)
        self.assertNotIn("internal_detail", summary)
        self.assertNotIn("_invocation_argv", summary)
        self.assertEqual(configuration["schema_version"], 1)
        self.assertEqual(
            configuration["invocation_arguments"],
            ["--dataset-name", "grapes"],
        )
        self.assertEqual(
            configuration["resolved_arguments"]["internal_detail"], {"value": 2}
        )
        self.assertEqual(
            configuration["runtime_configuration"]["General"]["device"], "cpu"
        )

    def test_partial_detector_weights_keep_current_module_mapping(self):
        """Partial detector loading targets the three current detector modules."""
        args = self._weight_args(
            load_partial_weights=True,
            load_bbox_det_weights=True,
            load_per_object_counting_weights=False,
            load_per_object_attributes_weights=False,
            network_type="bbox_detection",
            bbox_detection_weights_file="detector.pt",
        )
        model = mock.Mock()
        state_dict = object()
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.torch.load.assert_called_once_with(
            "detector.pt",
            map_location=self.model_setup.config.General.device,
        )
        model.load_state_dict.assert_not_called()
        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=["backbone_1", "find_1", "where"],
            strict=False,
            verbose=False,
        )

    def test_detector_only_mode_enters_weight_loading(self):
        """Detector-only initialization is not skipped when load_weights is false."""
        args = self._weight_args(
            load_weights=False,
            load_only_bbox_weights=True,
        )
        model = mock.Mock()

        with mock.patch.object(
            self.runner, "load_requested_weights"
        ) as load_requested_weights:
            self._run_through_weight_setup(args, model)

        load_requested_weights.assert_called_once_with(model, args)

    def test_full_counting_weights_keep_current_module_mapping(self):
        """Full counting-regression loading targets backbone and estimator."""
        args = self._weight_args(
            load_full_model_weights=True,
            full_model_weights="full.pt",
        )
        model = mock.Mock()
        state_dict = {"estimator.regSubmodel.weight": object()}
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=["backbone", "estimator"],
            strict=False,
            verbose=False,
        )

    def test_full_weight_key_check_prints_successful_comparison(self):
        """A successful full-checkpoint key check produces visible output."""
        args = self._weight_args(
            load_full_model_weights=True,
            full_model_weights="full.pt",
        )
        model = mock.Mock()
        model.state_dict.return_value = {
            "backbone.weight": object(),
            "estimator.weight": object(),
        }
        state_dict = {
            "backbone.weight": object(),
            "estimator.weight": object(),
        }
        self.model_setup.torch.load.return_value = state_dict
        self.model_setup.list_checkpoint_modules.side_effect = (
            lambda value: sorted({key.split(".")[0] for key in value})
        )

        with mock.patch("builtins.print") as print_mock:
            self.model_setup._load_full_weights(model, args)

        print_mock.assert_any_call(
            "Checkpoint modules:",
            ["backbone", "estimator"],
        )
        print_mock.assert_any_call(
            "Built model modules:",
            ["backbone", "estimator"],
        )
        print_mock.assert_any_call(
            "Key check passed: checkpoint modules match the built model.\n"
        )

    def test_partial_root_attributes_keep_current_module_mapping(self):
        """Root keypoint attributes retain their five-module partial mapping."""
        args = self._weight_args(
            load_partial_weights=True,
            load_bbox_det_weights=False,
            load_per_object_counting_weights=False,
            load_per_object_attributes_weights=True,
            network_type="per_object_attributes",
            estimate_type="withKeyPoints",
            per_object_weights_file="attributes.pt",
        )
        model = mock.Mock()
        state_dict = {
            "estimator_length.sec_reg_layer.weight": object(),
            "estimator_diameter.sec_reg_layer.weight": object(),
            "estimator_color.sec_reg_layer.weight": object(),
        }
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=[
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ],
            strict=False,
            verbose=False,
        )

    def test_full_regression_attributes_load_real_model_modules(self):
        """Combined regression weights target the model's actual child modules."""
        args = self._weight_args(
            load_full_model_weights=True,
            full_model_weights="full_attributes.pt",
            network_type="per_object_attributes",
            estimate_type="reg_fpn_p3_p7_min_sig",
        )
        model = mock.Mock()
        state_dict = {
            "estimator_length.regSubmodel.weight": object(),
            "estimator_diameter.regSubmodel.weight": object(),
            "estimator_color.regSubmodel.weight": object(),
        }
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=[
                "bbox_detection",
                "backbone_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ],
            strict=False,
            verbose=False,
        )

    def test_full_regression_attributes_reject_keypoint_estimators(self):
        """A keypoint checkpoint cannot initialize regression estimators."""
        args = self._weight_args(
            load_full_model_weights=True,
            full_model_weights="keypoint_attributes.pt",
            network_type="per_object_attributes",
            estimate_type="reg_fpn_p3_p7_min_sig",
        )
        model = mock.Mock()
        self.model_setup.torch.load.return_value = {
            "estimator_length.sec_reg_layer.weight": object(),
            "estimator_diameter.sec_reg_layer.weight": object(),
            "estimator_color.sec_reg_layer.weight": object(),
        }

        with self.assertRaisesRegex(
            ValueError,
            "appears to use 'withKeyPoints'",
        ):
            self.model_setup._load_full_weights(model, args)

        self.model_setup.load_submodule_weights.assert_not_called()

    def test_estimator_validation_rejects_mismatches_for_all_model_families(self):
        """Every estimator-bearing model rejects the opposite weight architecture."""
        keypoint_single = {"estimator.sec_reg_layer.weight": object()}
        regression_single = {"estimator.regSubmodel.weight": object()}
        keypoint_attributes = {
            f"estimator_{name}.sec_reg_layer.weight": object()
            for name in ("length", "diameter", "color")
        }
        regression_attributes = {
            f"estimator_{name}.regSubmodel.weight": object()
            for name in ("length", "diameter", "color")
        }
        cases = (
            (
                "per_image_estimation",
                "withKeyPoints",
                regression_single,
            ),
            (
                "per_image_estimation",
                "reg_fpn_p3_p7_min_sig",
                keypoint_single,
            ),
            ("per_object_counting", "withKeyPoints", regression_single),
            (
                "per_object_counting",
                "reg_fpn_p3_p7_min_sig",
                keypoint_single,
            ),
            (
                "per_object_attributes",
                "withKeyPoints",
                regression_attributes,
            ),
            (
                "per_object_attributes",
                "reg_fpn_p3_p7_min_sig",
                keypoint_attributes,
            ),
            (
                "per_object_attributes_multibranch",
                "withKeyPoints",
                regression_attributes,
            ),
        )

        for network_type, estimate_type, state_dict in cases:
            with self.subTest(
                network_type=network_type,
                estimate_type=estimate_type,
            ):
                args = SimpleNamespace(
                    network_type=network_type,
                    estimate_type=estimate_type,
                )
                with self.assertRaisesRegex(ValueError, "appears to use"):
                    self.model_setup._validate_estimator_weights(
                        state_dict,
                        args,
                        "Test checkpoint",
                    )

    def test_partial_regression_attributes_exclude_find_module(self):
        """Regression attributes load only modules used by the regression path."""
        args = self._weight_args(
            load_partial_weights=True,
            load_bbox_det_weights=False,
            load_per_object_counting_weights=False,
            load_per_object_attributes_weights=True,
            network_type="per_object_attributes",
            estimate_type="reg_fpn_p3_p7_min_sig",
            per_object_weights_file="attributes_reg.pt",
        )
        model = mock.Mock()
        state_dict = {
            "estimator_length.regSubmodel.weight": object(),
            "estimator_diameter.regSubmodel.weight": object(),
            "estimator_color.regSubmodel.weight": object(),
        }
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=[
                "backbone_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ],
            strict=False,
            verbose=False,
        )

    def test_partial_object_counting_keeps_current_module_mapping(self):
        """Grape keypoint counting retains its three-module partial mapping."""
        args = self._weight_args(
            load_partial_weights=True,
            load_bbox_det_weights=False,
            load_per_object_counting_weights=True,
            load_per_object_attributes_weights=False,
            network_type="per_object_counting",
            estimate_type="withKeyPoints",
            per_object_weights_file="counting.pt",
        )
        model = mock.Mock()
        state_dict = {"estimator.sec_reg_layer.weight": object()}
        self.model_setup.torch.load.return_value = state_dict

        self._run_through_weight_setup(args, model)

        self.model_setup.load_submodule_weights.assert_called_once_with(
            model,
            state_dict,
            submodule_names=["backbone_2", "find_2", "estimator"],
            strict=False,
            verbose=False,
        )

    def test_legacy_detector_export_returns_before_model_execution(self):
        """Legacy detector conversion saves one task and exits the runner."""
        args = self._weight_args(
            load_weights=False,
            save_from_model_file=True,
            network_type="bbox_detection",
            model_path="legacy.pt",
        )
        model = mock.Mock()
        legacy_model = mock.Mock()
        self.model_setup.torch.load.return_value = legacy_model

        self._run_through_weight_setup(args, model, stop_after_to=False)

        self.model_setup.save_partial_weights.assert_called_once_with(
            args,
            model,
            legacy_model,
            tasks=["bbox_detection"],
        )
        model.to.assert_not_called()

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

    def test_tee_omits_transient_download_percentages_from_results_stream(self):
        """Carriage-return download updates remain live but are not persisted."""
        console = mock.Mock()
        results_file = mock.Mock()
        tee = self.runner.Tee(
            console,
            results_file,
            suppress_transient_progress_after_first=True,
        )

        tee.write("\r 12.3%")
        tee.write("Download complete: 100% verified\n")

        self.assertEqual(
            [call.args[0] for call in console.write.call_args_list],
            ["\r 12.3%", "Download complete: 100% verified\n"],
        )
        results_file.write.assert_called_once_with(
            "Download complete: 100% verified\n"
        )

    def test_tee_omits_streamlit_progress_protocol_from_results_stream(self):
        """Structured GUI progress is displayed live but not persisted."""
        console = mock.Mock()
        results_file = mock.Mock()
        tee = self.runner.Tee(
            console,
            results_file,
            suppress_transient_progress_after_first=True,
        )
        progress = "__LEGONET_PROGRESS__\tLoading annotations:\t1\t25\n"

        tee.write(progress)

        console.write.assert_called_once_with(progress)
        results_file.write.assert_not_called()

    def test_tee_omits_separately_written_progress_newline_from_results(self):
        """A suppressed progress record cannot leave a blank saved line."""
        console = mock.Mock()
        results_file = mock.Mock()
        tee = self.runner.Tee(
            console,
            results_file,
            suppress_transient_progress_after_first=True,
        )

        progress = "__LEGONET_PROGRESS__\tLoading annotations:\t1\t25"
        tee.write(progress)
        tee.write("\n")
        tee.write("Detection metric scope: 25 images\n")

        self.assertEqual(
            [call.args[0] for call in console.write.call_args_list],
            [progress, "\n", "Detection metric scope: 25 images\n"],
        )
        results_file.write.assert_called_once_with(
            "Detection metric scope: 25 images\n"
        )

    def test_training_dispatch_receives_the_complete_data_bundle(self):
        """The runner delegates training with every constructed data object."""
        data = SimpleNamespace(
            dataset_train=object(),
            dataset_val=object(),
            sampler=object(),
            sampler_val=object(),
            dataloader_train=object(),
            dataloader_val=object(),
        )
        args = SimpleNamespace(
            load_weights=False,
            save_from_model_file=False,
            network_type="per_image_estimation",
            freeze_detection=False,
            run_script="Training",
        )
        built_model = mock.Mock()
        prepared_model = mock.Mock()
        built_model.to.return_value = prepared_model

        with mock.patch.object(
            self.runner, "build_data", return_value=data
        ), mock.patch.object(
            self.runner, "model_build", return_value=built_model
        ), mock.patch.object(self.runner, "train_model") as train_model:
            self.runner._run(args)

        train_model.assert_called_once_with(
            args,
            prepared_model,
            data.dataset_train,
            data.dataset_val,
            data.sampler,
            data.sampler_val,
            data.dataloader_train,
            data.dataloader_val,
        )

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
            network_type="per_object_counting",
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
