"""Tests for the named-module checkpoint export script."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import export_named_weights


class ExportNamedWeightsTests(unittest.TestCase):
    def test_export_loads_checkpoint_on_cpu_and_forwards_output_directory(self):
        model = object()
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_file = Path(temporary_dir) / "model.pt"
            checkpoint_file.touch()
            output_dir = Path(temporary_dir) / "exported"

            with (
                mock.patch.object(export_named_weights.torch, "load", return_value=model) as load,
                mock.patch.object(export_named_weights, "save_named_module_weights") as save,
            ):
                export_named_weights.export_named_weights(checkpoint_file, output_dir)

        load.assert_called_once_with(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        save.assert_called_once_with(model, str(output_dir))

    def test_missing_checkpoint_fails_before_loading(self):
        checkpoint_file = Path("missing-checkpoint.pt")

        with mock.patch.object(export_named_weights.torch, "load") as load:
            with self.assertRaisesRegex(FileNotFoundError, "missing-checkpoint.pt"):
                export_named_weights.export_named_weights(checkpoint_file, Path("output"))

        load.assert_not_called()

    def test_main_accepts_input_and_output_paths(self):
        with mock.patch.object(export_named_weights, "export_named_weights") as export:
            result = export_named_weights.main(
                [
                    "--checkpoint-file",
                    "model.pt",
                    "--output-dir",
                    "named-weights",
                ]
            )

        self.assertEqual(result, 0)
        export.assert_called_once_with(Path("model.pt"), Path("named-weights"))


if __name__ == "__main__":
    unittest.main()
