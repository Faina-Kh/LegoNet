"""Combine LegoNet detector and per-object partial checkpoints."""

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
    combine_partial_checkpoints,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse partial-checkpoint combination arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine validated detector and per-object partial checkpoints "
            "into one full-model checkpoint."
        )
    )
    parser.add_argument("--detector-weights-file", required=True)
    parser.add_argument("--per-object-weights-file", required=True)
    parser.add_argument("--full-output-file", required=True)
    parser.add_argument(
        "--network-type",
        required=True,
        choices=PER_OBJECT_NETWORKS,
    )
    parser.add_argument(
        "--estimate-type",
        required=True,
        choices=ESTIMATE_TYPE_CHOICES,
    )
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run partial-checkpoint combination."""
    args = parse_args(arguments)
    combine_partial_checkpoints(
        detector_weights_file=args.detector_weights_file,
        per_object_weights_file=args.per_object_weights_file,
        full_output_file=args.full_output_file,
        network_type=args.network_type,
        estimate_type=args.estimate_type,
        attribute_names=args.attribute_names,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
