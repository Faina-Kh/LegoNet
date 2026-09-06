"""Tests for adapting the published Four Crops layouts to the TRL loader."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from legonet.four_crops import resolve_four_crops_split


def _write_pair(directory: Path, stem: str, rows: list[tuple[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{stem}_TRL.csv").open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows((name, trl) for name, trl in rows)
    with (directory / f"{stem}_pointsOutput.csv").open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(([name] if trl == "0" else [name, "1", "2"]) for name, trl in rows)
    for name, _ in rows:
        (directory / name).write_bytes(b"image")


def test_dataset_1_resolves_published_split_and_empty_points(tmp_path: Path) -> None:
    split = tmp_path / "sub_Test"
    _write_pair(split, "Test", [("empty.jpg", "0"), ("root.jpg", "12.5")])

    result = resolve_four_crops_split(tmp_path, "dataset_1", "Test")

    assert result.trl_file == split / "Test_TRL.csv"
    assert result.points_file == split / "Test_pointsOutput.csv"
    assert result.base_dir == split


def test_dataset_3_uses_root_annotations_and_is_inference_only(tmp_path: Path) -> None:
    _write_pair(tmp_path, "published", [("root.jpg", "2")])
    (tmp_path / "published_TRL.csv").rename(tmp_path / "TRL.csv")
    (tmp_path / "published_pointsOutput.csv").rename(tmp_path / "pointsOutput.csv")

    result = resolve_four_crops_split(tmp_path, "dataset_3", "Test")
    assert result.base_dir == tmp_path
    with pytest.raises(ValueError, match="inference-only"):
        resolve_four_crops_split(tmp_path, "dataset_3", "Train")


def test_dataset_4_combines_acquisitions_with_relative_image_paths(tmp_path: Path) -> None:
    for folder, name, trl in (("CORN 2020_tube 16", "corn.jpg", "3"), ("MELON 2018_tube 17", "melon.jpg", "0")):
        acquisition = tmp_path / folder
        _write_pair(acquisition, "published", [(name, trl)])
        (acquisition / "published_TRL.csv").rename(acquisition / "TRL.csv")
        (acquisition / "published_pointsOutput.csv").rename(acquisition / "pointsOutput.csv")

    result = resolve_four_crops_split(tmp_path, "dataset_4", "Test", tmp_path / "manifests")

    assert result.base_dir == tmp_path
    assert result.trl_file.read_text(encoding="utf-8").splitlines() == [
        "CORN 2020_tube 16/corn.jpg,3",
        "MELON 2018_tube 17/melon.jpg,0",
    ]
    assert result.points_file.read_text(encoding="utf-8").splitlines()[-1] == "MELON 2018_tube 17/melon.jpg"


def test_rejects_mismatched_annotation_images(tmp_path: Path) -> None:
    split = tmp_path / "sub_Test"
    _write_pair(split, "Test", [("root.jpg", "2")])
    (split / "Test_pointsOutput.csv").write_text("different.jpg,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image sets differ"):
        resolve_four_crops_split(tmp_path, "dataset_2", "Test")


def test_rejects_missing_referenced_image(tmp_path: Path) -> None:
    split = tmp_path / "sub_Test"
    _write_pair(split, "Test", [("root.jpg", "2")])
    (split / "root.jpg").unlink()

    with pytest.raises(ValueError, match="is missing"):
        resolve_four_crops_split(tmp_path, "dataset_1", "Test")


def test_accepts_published_negative_boundary_coordinates(tmp_path: Path) -> None:
    split = tmp_path / "sub_Test"
    _write_pair(split, "Test", [("root.jpg", "2")])
    (split / "Test_pointsOutput.csv").write_text(
        "root.jpg,-1,12,8,-2\n",
        encoding="utf-8",
    )

    result = resolve_four_crops_split(tmp_path, "dataset_1", "Test")

    assert result.points_file == split / "Test_pointsOutput.csv"
