"""Tests for Dataset 4 per-acquisition TRL reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from legonet.eval.per_image_attribute_eval import (
    _dataset_4_subfolder,
    write_dataset_4_subfolder_metrics,
)


def test_dataset_4_manifest_path_identifies_acquisition_folder() -> None:
    args = SimpleNamespace(
        dataset_name="roots_four_crops",
        dataset_subset="dataset_4",
    )

    assert _dataset_4_subfolder("CORN 2020_tube 16/root.jpg", args) == "CORN 2020_tube 16"


def test_writes_separate_metrics_for_each_dataset_4_subfolder(tmp_path: Path) -> None:
    groups = {
        "CORN": {
            "ground_truth": [10.0, 20.0],
            "predictions": [8.0, 22.0],
            "point_truth": [1.0, 0.0],
            "point_scores": [0.9, 0.4],
        },
        "MELON": {
            "ground_truth": [5.0],
            "predictions": [4.0],
            "point_truth": [1.0],
            "point_scores": [0.8],
        },
    }

    output, summary = write_dataset_4_subfolder_metrics(
        groups,
        tmp_path,
        include_point_ap=True,
    )

    with output.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["subfolder"] for row in rows] == ["CORN", "MELON"]
    assert rows[0]["images"] == "2"
    assert rows[0]["mean_relative_deviation_gt_positive"] == "0.15000000000000002"
    assert rows[1]["point_mAP"] == "1.0"
    assert "CORN: images=2" in summary
    assert "MELON: images=1" in summary
