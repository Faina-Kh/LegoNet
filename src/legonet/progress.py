"""Small progress-display helpers shared by command-line evaluation code."""

from __future__ import annotations

import sys
from typing import TextIO


def print_image_progress(
    label: str,
    current: int,
    total: int,
    stream: TextIO | None = None,
) -> None:
    """Update one terminal line, or emit one final line for redirected output."""
    output = sys.stdout if stream is None else stream
    message = f"{label} {current}/{total}"
    complete = current >= total

    if output.isatty():
        print(
            f"\r{message}",
            end="\n" if complete else "",
            file=output,
            flush=True,
        )
    elif complete:
        print(message, file=output, flush=True)
