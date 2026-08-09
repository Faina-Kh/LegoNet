"""Remove one named module from a LegoNet checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.checkpoint_conversion import remove_checkpoint_module


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint module-removal arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove a named module and all of its parameters from a state-dict "
            "checkpoint. The source checkpoint is never modified."
        )
    )
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--module-name", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run checkpoint module removal."""
    args = parse_args(arguments)
    remove_checkpoint_module(
        weights_file=args.weights_file,
        output_file=args.output_file,
        module_name=args.module_name,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
