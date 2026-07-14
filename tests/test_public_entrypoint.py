"""Subprocess smoke tests for the public LegoNet command-line script."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCRIPT = PROJECT_ROOT / "scripts" / "run_legonet.py"


def run_public_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the public entry point in a clean subprocess environment."""
    environment = os.environ.copy()
    environment.pop("LEGONET_STORAGE_PATH", None)
    return subprocess.run(
        [sys.executable, str(PUBLIC_SCRIPT), *arguments],
        cwd=str(PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class PublicEntryPointSmokeTests(unittest.TestCase):
    """Verify lightweight public CLI behavior without ML dependencies."""

    def test_help_succeeds(self) -> None:
        """The public command exposes help without starting an experiment."""
        result = run_public_script("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--storage-path", result.stdout)
        self.assertIn("--network-type", result.stdout)

    def test_missing_storage_path_fails_cleanly(self) -> None:
        """Missing public storage configuration returns an actionable error."""
        result = run_public_script()

        self.assertEqual(result.returncode, 2)
        self.assertIn("Configuration error", result.stderr)
        self.assertIn("LEGONET_STORAGE_PATH", result.stderr)

    def test_invalid_dataset_network_combination_fails_cleanly(self) -> None:
        """The public script rejects unsupported combinations before ML setup."""
        with TemporaryDirectory() as storage_path:
            result = run_public_script(
                "--storage-path",
                storage_path,
                "--dataset-name",
                "grapes",
                "--network-type",
                "per_object_attributes",
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("not supported for dataset", result.stderr)


if __name__ == "__main__":
    unittest.main()
