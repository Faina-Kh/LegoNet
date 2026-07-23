"""Tests for safe checkpoint replacement."""

import importlib
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


class CheckpointingTests(unittest.TestCase):
    """Verify checkpoint cleanup is narrow and occurs after saving."""

    @classmethod
    def setUpClass(cls):
        """Import checkpointing with a lightweight Torch replacement."""
        torch_module = types.ModuleType("torch")
        torch_module.save = mock.Mock()
        sys.modules.pop("legonet.checkpointing", None)
        sys.modules.pop("legonet.manage_weights", None)
        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            cls.checkpointing = importlib.import_module("legonet.checkpointing")
            cls.manage_weights = importlib.import_module("legonet.manage_weights")

    def setUp(self):
        """Reset the mocked Torch save operation."""
        self.checkpointing.torch.save.reset_mock()
        self.checkpointing.torch.save.side_effect = None

    def test_best_checkpoint_replaces_only_legonet_checkpoints(self):
        """Replacing a best checkpoint preserves unrelated directory contents."""
        model = mock.Mock()
        model.state_dict.return_value = {"weight": 1}

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            old_checkpoint = directory / "legonet_epoch=2.pt"
            unrelated_weights = directory / "detector.pt"
            notes = directory / "notes.txt"
            old_checkpoint.write_text("old")
            unrelated_weights.write_text("keep")
            notes.write_text("keep")

            path = self.checkpointing.save_epoch_checkpoint(
                model,
                epoch=3,
                replace_existing=True,
                weights_dir=temporary_directory,
            )

            self.assertEqual(path, directory / "legonet_epoch=3.pt")
            self.assertFalse(old_checkpoint.exists())
            self.assertTrue(unrelated_weights.exists())
            self.assertTrue(notes.exists())
            self.checkpointing.torch.save.assert_called_once_with(
                {"weight": 1},
                str(path),
            )

    def test_failed_save_does_not_remove_previous_checkpoint(self):
        """A save failure leaves the previous valid checkpoint untouched."""
        model = mock.Mock()
        self.checkpointing.torch.save.side_effect = RuntimeError("disk full")

        with tempfile.TemporaryDirectory() as temporary_directory:
            old_checkpoint = Path(temporary_directory) / "legonet_epoch=2.pt"
            old_checkpoint.write_text("old")

            with self.assertRaises(RuntimeError):
                self.checkpointing.save_epoch_checkpoint(
                    model,
                    epoch=3,
                    replace_existing=True,
                    weights_dir=temporary_directory,
                )

            self.assertTrue(old_checkpoint.exists())

    def test_checkpoint_with_different_head_modules_is_rejected(self):
        """A keypoint head cannot be loaded into a regression architecture."""
        state_dict = {
            "backbone_2.layer.weight": object(),
            "find_2.classifier.weight": object(),
            "estimator.regressor.weight": object(),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"unexpected modules: \['find_2'\]",
        ):
            self.manage_weights.validate_checkpoint_modules(
                state_dict,
                ["backbone_2", "estimator"],
                "Per-object counting checkpoint",
            )

    def test_checkpoint_with_matching_modules_is_accepted(self):
        """Matching module sets pass architecture validation."""
        state_dict = {
            "backbone_2.layer.weight": object(),
            "estimator.regressor.weight": object(),
        }

        self.manage_weights.validate_checkpoint_modules(
            state_dict,
            ["backbone_2", "estimator"],
            "Per-object counting checkpoint",
        )

    def test_module_listing_marks_modules_without_weights(self):
        """Structural detector helpers are identified as weightless modules."""
        weighted = mock.Mock()
        weighted.state_dict.return_value = {"weight": object()}
        weightless = mock.Mock()
        weightless.state_dict.return_value = {}
        model = mock.Mock()
        model.named_children.return_value = [
            ("backbone_1", weighted),
            ("anchors", weightless),
        ]
        output = io.StringIO()

        with redirect_stdout(output):
            self.manage_weights.print_module_names(model)

        self.assertIn("backbone_1", output.getvalue())
        self.assertIn("anchors (no weights)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
