"""Tests for the import-safe LegoNet command-line entry point."""

import importlib
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


class MainEntryPointTests(unittest.TestCase):
    """Verify import safety and explicit runner dispatch."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the entry point with a lightweight PyTorch stub."""
        torch_stub = types.ModuleType("torch")
        cls.runner_stub = types.ModuleType("legonet.runner")
        cls.runner_stub.run = mock.Mock()

        sys.modules.pop("legonet.cli", None)
        with mock.patch.dict(
            sys.modules,
            {"torch": torch_stub, "legonet.runner": cls.runner_stub},
        ):
            cls.main_module = importlib.import_module("legonet.cli")

    def setUp(self) -> None:
        """Reset the runner mock between tests."""
        self.runner_stub.run.reset_mock()

    def test_import_does_not_start_runner(self) -> None:
        """Importing the module must not launch training or inference."""
        self.runner_stub.run.assert_not_called()

    def test_main_configures_arguments_and_runs_once(self) -> None:
        """Calling main explicitly configures and dispatches one run."""
        import legonet

        parsed_args = SimpleNamespace()
        configured_args = SimpleNamespace(txt_results="")

        with ExitStack() as stack:
            parse_args = stack.enter_context(
                mock.patch.object(
                    self.main_module, "parse_args", return_value=parsed_args
                )
            )
            configure_runtime = stack.enter_context(
                mock.patch.object(
                    self.main_module,
                    "configure_runtime",
                    return_value=configured_args,
                )
            )
            print_to_csv = stack.enter_context(
                mock.patch.object(self.main_module, "print_to_csv")
            )
            stack.enter_context(
                mock.patch.object(legonet, "runner", self.runner_stub, create=True)
            )
            stack.enter_context(
                mock.patch.dict(sys.modules, {"legonet.runner": self.runner_stub})
            )
            stack.enter_context(mock.patch("builtins.print"))

            result = self.main_module.main(["--help-placeholder"])

        self.assertEqual(result, 0)
        parse_args.assert_called_once_with(["--help-placeholder"])
        configure_runtime.assert_called_once_with(parsed_args)
        self.runner_stub.run.assert_called_once_with(configured_args)
        print_to_csv.assert_called_once()

    def test_cli_storage_path_takes_precedence(self) -> None:
        """An explicit CLI path overrides the environment setting."""
        with TemporaryDirectory() as cli_dir, TemporaryDirectory() as env_dir:
            result = self.main_module.resolve_storage_path(
                cli_dir,
                {"LEGONET_STORAGE_PATH": env_dir},
            )

        self.assertEqual(result, str(Path(cli_dir)))

    def test_storage_path_can_come_from_environment(self) -> None:
        """The environment variable supplies the root when CLI input is absent."""
        with TemporaryDirectory(prefix="legonet storage ") as storage_dir:
            result = self.main_module.resolve_storage_path(
                None,
                {"LEGONET_STORAGE_PATH": storage_dir},
            )

        self.assertEqual(result, str(Path(storage_dir)))

    def test_missing_storage_path_is_rejected(self) -> None:
        """Missing CLI and environment values produce an actionable error."""
        with self.assertRaisesRegex(ValueError, "--storage-path"):
            self.main_module.resolve_storage_path(None, {})

    def test_nonexistent_storage_path_is_rejected(self) -> None:
        """The configured root must already exist as a directory."""
        with TemporaryDirectory() as parent:
            missing_path = Path(parent) / "missing"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                self.main_module.resolve_storage_path(str(missing_path), {})

    def test_main_reports_configuration_error_without_running(self) -> None:
        """Invalid public input fails cleanly before runner dispatch."""
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(self.main_module, "parse_args", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    self.main_module,
                    "configure_runtime",
                    side_effect=ValueError("missing storage"),
                )
            )
            print_mock = stack.enter_context(
                mock.patch.object(self.main_module, "print", create=True)
            )
            result = self.main_module.main([])

        self.assertEqual(result, 2)
        self.runner_stub.run.assert_not_called()
        self.assertTrue(
            any(
                call[0] and "Configuration error" in call[0][0]
                for call in print_mock.call_args_list
            )
        )

    def test_explicit_false_boolean_values_are_preserved(self) -> None:
        """False values from CLI and Streamlit survive default resolution."""
        args = self.main_module.parse_args(
            [
                "--have-gt",
                "false",
                "--to-draw",
                "false",
                "--evaluate-detection",
                "false",
                "--load-weights",
                "false",
                "--save-from-model-file",
                "false",
            ]
        )

        result = self.main_module.resolve_boolean_options(args)

        self.assertFalse(result.have_GT)
        self.assertFalse(result.to_draw)
        self.assertFalse(result.evaluate_detection)
        self.assertFalse(result.load_weights)
        self.assertFalse(result.save_from_model_file)

    def test_boolean_defaults_preserve_legacy_run_mode(self) -> None:
        """Omitted options still default to GT evaluation and weight loading."""
        result = self.main_module.resolve_boolean_options(
            self.main_module.parse_args([])
        )

        self.assertTrue(result.have_GT)
        self.assertFalse(result.to_draw)
        self.assertTrue(result.evaluate_detection)
        self.assertTrue(result.load_weights)
        self.assertFalse(result.save_from_model_file)

    def test_legacy_export_can_be_selected_explicitly(self) -> None:
        """Disabling loading permits explicit legacy weight export."""
        args = self.main_module.parse_args(
            [
                "--load-weights",
                "false",
                "--save-from-model-file",
                "true",
            ]
        )

        result = self.main_module.resolve_boolean_options(args)

        self.assertFalse(result.load_weights)
        self.assertTrue(result.save_from_model_file)

    def test_loading_and_legacy_export_are_mutually_exclusive(self) -> None:
        """Contradictory weight modes fail before filesystem setup."""
        args = self.main_module.parse_args(
            [
                "--load-weights",
                "true",
                "--save-from-model-file",
                "true",
            ]
        )

        with self.assertRaisesRegex(ValueError, "cannot both be true"):
            self.main_module.resolve_boolean_options(args)

    def test_supported_configuration_is_accepted(self) -> None:
        """A documented roots attribute-training configuration is valid."""
        args = SimpleNamespace(
            dataset_name="roots",
            network_type="per_object_attributes",
            estimate_type="reg_fpn_p3_p7_min_sig",
            run_script="Training",
            val_set="Val",
            have_GT=True,
        )

        self.assertIs(self.main_module.validate_configuration(args), args)

    def test_roots_per_image_runtime_flags_are_initialized(self) -> None:
        """Per-image roots runs receive the dataset-wide inference flags."""
        args = SimpleNamespace(
            dataset_name="roots",
            network_type="per_image_estimation_keypoints",
        )

        result = self.main_module.initialize_dataset_runtime_flags(args)

        self.assertTrue(result.predict_empty_image)
        self.assertFalse(result.do_nmcs)

    def test_load_only_bbox_weights_option_is_parsed(self) -> None:
        """Per-object heads can be initialized while loading only the detector."""
        args = self.main_module.parse_args(
            ["--load-only-bbox-weights", "true"]
        )

        self.assertTrue(args.load_only_bbox_weights)

    def test_explicit_detector_only_mode_overrides_legacy_loading_flags(self) -> None:
        """The explicit mode is the source of truth for checkpoint loading."""
        args = self.main_module.parse_args(
            ["--weights-mode", "detector_only"]
        )
        args.run_script = "Training"
        args.network_type = "per_object_counting"
        self.main_module.resolve_boolean_options(args)

        result = self.main_module.configure_weights_mode(args)

        self.assertEqual(result.weights_mode, "detector_only")
        self.assertTrue(result.load_only_bbox_weights)
        self.assertFalse(result.load_weights)
        self.assertEqual(result.weights_type, "partial_weights")

    def test_explicit_weights_file_is_resolved(self) -> None:
        """User-supplied checkpoint paths are validated as files."""
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()

            result = self.main_module.require_weights_file(
                str(checkpoint), "--full-weights-file"
            )

        self.assertEqual(result, str(checkpoint.resolve()))

    def test_missing_explicit_weights_file_is_rejected(self) -> None:
        """A selected loading mode cannot silently fall back to another file."""
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.main_module.require_weights_file(
                "missing-model.pt", "--full-weights-file"
            )

    def test_load_only_bbox_weights_is_rejected_for_inference(self) -> None:
        """Detector-only initialization is a per-object training operation."""
        args = SimpleNamespace(
            dataset_name="grapes",
            network_type="per_object_counting",
            estimate_type="reg_fpn_p3_p7_min_sig",
            run_script="Inference",
            val_set="Test",
            have_GT=True,
            load_only_bbox_weights=True,
        )

        with self.assertRaisesRegex(ValueError, "only when training"):
            self.main_module.validate_configuration(args)

    def test_load_only_bbox_weights_is_rejected_for_detector_training(self) -> None:
        """Detection-only training does not initialize a per-object head."""
        args = SimpleNamespace(
            dataset_name="grapes",
            network_type="bbox_detection",
            estimate_type="withKeyPoints",
            run_script="Training",
            val_set="Val",
            have_GT=True,
            load_only_bbox_weights=True,
        )

        with self.assertRaisesRegex(ValueError, "per-object network"):
            self.main_module.validate_configuration(args)

    def test_dataset_rejects_an_incompatible_network(self) -> None:
        """Dataset-specific model choices fail with the supported alternatives."""
        args = SimpleNamespace(
            dataset_name="grapes",
            network_type="per_object_attributes",
            estimate_type="withKeyPoints",
            run_script="Inference",
            val_set="Test",
            have_GT=True,
        )

        with self.assertRaisesRegex(ValueError, "not supported for dataset"):
            self.main_module.validate_configuration(args)

    def test_network_rejects_an_incompatible_estimator(self) -> None:
        """Fixed-estimator networks report their supported estimate type."""
        args = SimpleNamespace(
            dataset_name="roots",
            network_type="per_image_estimation_keypoints",
            estimate_type="reg_fpn_p3_p7_min_sig",
            run_script="Inference",
            val_set="Test",
            have_GT=True,
        )

        with self.assertRaisesRegex(ValueError, "not supported for network"):
            self.main_module.validate_configuration(args)

    def test_multibranch_network_requires_keypoints(self) -> None:
        """The multibranch architecture remains keypoint-only."""
        args = SimpleNamespace(
            dataset_name="roots",
            network_type="per_object_attributes_multibranch",
            estimate_type="reg_fpn_p3_p7_min_sig",
            run_script="Inference",
            val_set="Test",
            have_GT=True,
        )

        with self.assertRaisesRegex(ValueError, "withKeyPoints"):
            self.main_module.validate_configuration(args)

    def test_training_requires_ground_truth(self) -> None:
        """Training without GT is rejected before data construction."""
        args = SimpleNamespace(
            dataset_name="grapes",
            network_type="per_object_counting",
            estimate_type="withKeyPoints",
            run_script="Training",
            val_set="Val",
            have_GT=False,
        )

        with self.assertRaisesRegex(ValueError, "requires --have-gt true"):
            self.main_module.validate_configuration(args)

    def test_training_requires_validation_split(self) -> None:
        """Training cannot accidentally use the held-out test split."""
        args = SimpleNamespace(
            dataset_name="roots",
            network_type="per_object_attributes",
            estimate_type="withKeyPoints",
            run_script="Training",
            val_set="Test",
            have_GT=True,
        )

        with self.assertRaisesRegex(ValueError, "requires --val-set Val"):
            self.main_module.validate_configuration(args)


if __name__ == "__main__":
    unittest.main()
