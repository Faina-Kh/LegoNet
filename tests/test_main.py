"""Tests for the import-safe LegoNet command-line entry point."""

import importlib
import sys
import types
import unittest
from contextlib import ExitStack
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


if __name__ == "__main__":
    unittest.main()
