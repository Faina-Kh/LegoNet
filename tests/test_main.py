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

        sys.modules.pop("legonet.scripts.main", None)
        with mock.patch.dict(
            sys.modules,
            {"torch": torch_stub, "legonet.runner": cls.runner_stub},
        ):
            cls.main_module = importlib.import_module("legonet.scripts.main")

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


if __name__ == "__main__":
    unittest.main()
