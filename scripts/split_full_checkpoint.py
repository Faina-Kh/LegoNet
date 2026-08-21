"""Split a current LegoNet full checkpoint into task-specific weight files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.checkpoint_conversion import (
    ESTIMATE_TYPE_CHOICES,
    PER_OBJECT_NETWORKS,
    convert_full_checkpoint,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint conversion arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split a current LegoNet full-model state dictionary into detector "
            "and per-object partial checkpoints."
        )
    )
    parser.add_argument("--full-weights-file", required=True)
    parser.add_argument(
        "--network-type",
        required=True,
        choices=PER_OBJECT_NETWORKS,
    )
    parser.add_argument("--estimate-type", choices=ESTIMATE_TYPE_CHOICES)
    parser.add_argument(
        "--attribute-names",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "Names used by estimator_<name> modules. Leave empty for counting; "
            "attributes networks require names."
        ),
    )
    parser.add_argument(
        "--detector-output-file",
        help="Optional. Omit to save only the per-object head weights.",
    )
    parser.add_argument("--per-object-output-file")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run checkpoint conversion."""
    args = parse_args(arguments)
    if args.estimate_type is None:
        raise SystemExit("--estimate-type is required.")
    if not args.per_object_output_file:
        raise SystemExit(
            "--per-object-output-file is required."
        )
    convert_full_checkpoint(
        full_weights_file=args.full_weights_file,
        detector_output_file=args.detector_output_file,
        per_object_output_file=args.per_object_output_file,
        network_type=args.network_type,
        estimate_type=args.estimate_type,
        attribute_names=args.attribute_names,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
