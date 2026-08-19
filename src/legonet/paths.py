"""Construct dataset and experiment paths below the configured storage root."""

from __future__ import annotations

import os
from pathlib import Path


def get_paths(storage_path: str, dataset_name: str) -> dict[str, str]:
    """Return and create the dataset and experiment directories for a dataset."""
    paths: dict[str, str] = {}

    if dataset_name == "grapes":
        datasets_path = os.path.join(
            storage_path,
            "Datasets",
            "Embrapa WGISD",
        )
    elif dataset_name == "roots":
        datasets_path = os.path.join(
            storage_path,
            "Datasets",
            "Grapevines data",
        )

    experiment_results_path = os.path.join(
        storage_path,
        "ExpResults",
        dataset_name,
    )

    paths["DATASETS_PATH"] = datasets_path
    paths["EXP_RESULTS_PATH"] = experiment_results_path

    for directory in paths.values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    return paths
