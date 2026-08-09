"""Export each named LegoNet module from a full model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.utils import save_named_module_weights


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint-export arguments."""
    parser = argparse.ArgumentParser(
        description="Export named LegoNet module weights as NumPy files."
    )
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        type=Path,
        help="Full-model checkpoint containing the serialized LegoNet model.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which module-specific weight folders are created.",
    )
    return parser.parse_args(arguments)


def export_named_weights(checkpoint_file: Path, output_dir: Path) -> None:
    """Load a full model checkpoint on CPU and export its named modules."""
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_file}")

    model = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    save_named_module_weights(model, str(output_dir))


def main(arguments: list[str] | None = None) -> int:
    """Run named-module checkpoint export."""
    args = parse_args(arguments)
    export_named_weights(args.checkpoint_file, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
