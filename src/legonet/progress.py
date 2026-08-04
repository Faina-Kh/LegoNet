"""Small progress-display helpers shared by command-line evaluation code."""

from __future__ import annotations

import os
import sys
from typing import TextIO


PROGRESS_PREFIX = "__LEGONET_PROGRESS__"


def _progress_message(label: str, current: int, total: int) -> str:
    """Format a compact text progress bar and count."""
    ratio = min(max(current / total, 0.0), 1.0) if total else 1.0
    completed = round(30 * ratio)
    bar = "#" * completed + "-" * (30 - completed)
    return f"{label} [{bar}] {current}/{total}"


def print_image_progress(
    label: str,
    current: int,
    total: int,
    stream: TextIO | None = None,
) -> None:
    """Update one terminal line, or emit one final line for redirected output."""
    output = sys.stdout if stream is None else stream
    message = _progress_message(label, current, total)
    complete = current >= total

    is_terminal = getattr(output, "isatty", lambda: False)()
    if is_terminal:
        print(
            f"\r{message}",
            end="\n" if complete else "",
            file=output,
            flush=True,
        )
    elif os.environ.get("LEGONET_PROGRESS_PROTOCOL") == "1":
        print(
            f"{PROGRESS_PREFIX}\t{label}\t{current}\t{total}",
            file=output,
            flush=True,
        )
    elif complete:
        print(message, file=output, flush=True)
