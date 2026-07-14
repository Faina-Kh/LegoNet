"""Tests for the editable IDE debugging entry point."""

from __future__ import annotations

import unittest

from scripts.debug_legonet import build_cli_arguments


class DebugEntryPointTests(unittest.TestCase):
    """Verify debug settings are translated without starting an experiment."""

    def test_build_cli_arguments_converts_names_values_and_booleans(self) -> None:
        """Settings use public option names and argparse-compatible values."""
        arguments = build_cli_arguments(
            {
                "storage_path": r"C:\LegoNet Storage",
                "num_of_epochs": 2,
                "have_gt": True,
                "to_draw": False,
                "unused": None,
            }
        )

        self.assertEqual(
            arguments,
            [
                "--storage-path",
                r"C:\LegoNet Storage",
                "--num-of-epochs",
                "2",
                "--have-gt",
                "true",
                "--to-draw",
                "false",
            ],
        )


if __name__ == "__main__":
    unittest.main()
