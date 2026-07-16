"""Editable same-process entry point for debugging LegoNet in an IDE."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legonet.cli import main


DebugValue = str | int | bool | None

# Edit these values for the experiment you want to debug. Keeping the storage
# path in LEGONET_STORAGE_PATH avoids committing a machine-specific path.
DEBUG_SETTINGS: dict[str, DebugValue] = {
    "storage_path": "C:\\Users\\bordezki\\Desktop\\LegoNet", # os.environ.get("LEGONET_STORAGE_PATH", ""),
    "dataset_name": "grapes",  # "roots" or "grapes"
    "network_type": "per_object_counting", #"bbox_detection", "per_object_counting", "per_object_attributes", "per_object_attributes_multibranch"
    "run_script": "Training",  # "Inference" or "Training"
    "val_set": "Val",  # "Test" or "Val"
    "gpu_num": "1",
    "current_results_dir": "debug_run_"+"grapes_per_object_counting_b",
    "estimate_type": "reg_fpn_p3_p7_min_sig", #"reg_fpn_p3_p7_min_sig", "withKeyPoints"
    "num_of_epochs": 300,
    "have_gt": True,
    "to_draw": False,
    "evaluate_detection": True,
    "load_weights": False,
    "load_only_bbox_weights": True,
    "weights_type": "partial_weights",
}


def build_cli_arguments(settings: Mapping[str, DebugValue]) -> list[str]:
    """Convert debug settings to the arguments accepted by the public CLI."""
    arguments: list[str] = []
    for name, value in settings.items():
        if value is None:
            continue

        text_value = str(value).lower() if isinstance(value, bool) else str(value)
        arguments.extend((f"--{name.replace('_', '-')}", text_value))

    return arguments


if __name__ == "__main__":
    raise SystemExit(main(build_cli_arguments(DEBUG_SETTINGS)))
