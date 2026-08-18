"""Tests for concise image-progress output."""

from __future__ import annotations

import io
import os
import unittest
from unittest import mock

from legonet.progress import print_image_progress


class _TerminalBuffer(io.StringIO):
    """String buffer that behaves like an interactive terminal."""

    def isatty(self) -> bool:
        """Report interactive-terminal behavior."""
        return True


class _TeeLikeStream:
    """Minimal redirected stream without an ``isatty`` method."""

    def __init__(self) -> None:
        self.value = ""

    def write(self, value: str) -> int:
        """Collect written output."""
        self.value += value
        return len(value)

    def flush(self) -> None:
        """Support the file-like interface used by ``print``."""


class ProgressTests(unittest.TestCase):
    """Verify terminal and redirected progress rendering."""

    def test_redirected_output_prints_only_the_final_count(self) -> None:
        """Streamlit-style pipes receive one completed progress line."""
        output = io.StringIO()

        for current in range(1, 4):
            print_image_progress("Loading images:", current, 3, output)

        self.assertEqual(
            output.getvalue(),
            "Loading images: [##############################] 3/3\n",
        )

    def test_stream_without_isatty_is_treated_as_redirected(self) -> None:
        """Custom Tee streams receive only the completed progress line."""
        output = _TeeLikeStream()

        print_image_progress("Loading images:", 1, 2, output)
        print_image_progress("Loading images:", 2, 2, output)

        self.assertEqual(
            output.value,
            "Loading images: [##############################] 2/2\n",
        )

    def test_streamlit_protocol_emits_each_progress_update(self) -> None:
        """Streamlit receives machine-readable updates for every image."""
        output = io.StringIO()

        with mock.patch.dict(os.environ, {"LEGONET_PROGRESS_PROTOCOL": "1"}):
            print_image_progress("Loading images:", 1, 2, output)
            print_image_progress("Loading images:", 2, 2, output)

        self.assertEqual(
            output.getvalue(),
            "__LEGONET_PROGRESS__\tLoading images:\t1\t2\n"
            "__LEGONET_PROGRESS__\tLoading images:\t2\t2\n",
        )

    def test_terminal_output_updates_with_carriage_returns(self) -> None:
        """Interactive shells update in place and finish with a newline."""
        output = _TerminalBuffer()

        print_image_progress("Loading images:", 1, 2, output)
        print_image_progress("Loading images:", 2, 2, output)

        self.assertEqual(
            output.getvalue(),
            "\rLoading images: [###############---------------] 1/2"
            "\rLoading images: [##############################] 2/2\n",
        )


if __name__ == "__main__":
    unittest.main()
