"""Resolve and validate the published Four Crops annotation layouts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class FourCropsSplit:
    """Annotation files and image root consumed by ``csv_LCCDataset``."""

    trl_file: Path
    points_file: Path
    base_dir: Path


def _safe_image_name(value: str, source: Path, line: int) -> PurePosixPath:
    """Validate and normalize a relative image name from a published CSV."""
    normalized = value.strip().replace("\\", "/")
    image = PurePosixPath(normalized)
    if not normalized or image.is_absolute() or ".." in image.parts:
        raise ValueError(f"Unsafe image path in {source} line {line}: {value!r}")
    return image


def _read_trl(path: Path, image_root: Path) -> list[tuple[PurePosixPath, str]]:
    """Read a TRL manifest and validate values and referenced images."""
    if not path.is_file():
        raise ValueError(f"Missing Four Crops TRL annotations: {path}")
    rows: list[tuple[PurePosixPath, str]] = []
    seen: set[PurePosixPath] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for line, row in enumerate(csv.reader(input_file), start=1):
            if len(row) != 2:
                raise ValueError(f"Expected image,TRL in {path} line {line}.")
            image = _safe_image_name(row[0], path, line)
            try:
                trl = float(row[1])
            except ValueError as error:
                raise ValueError(f"Invalid TRL in {path} line {line}: {row[1]!r}") from error
            if not math.isfinite(trl) or trl < 0:
                raise ValueError(f"TRL must be finite and nonnegative in {path} line {line}.")
            if image in seen:
                raise ValueError(f"Duplicate image in {path} line {line}: {image}")
            if not (image_root / Path(*image.parts)).is_file():
                raise ValueError(f"Four Crops image referenced by {path} is missing: {image}")
            seen.add(image)
            rows.append((image, row[1].strip()))
    if not rows:
        raise ValueError(f"Four Crops TRL annotations are empty: {path}")
    return rows


def _read_points(path: Path) -> dict[PurePosixPath, list[str]]:
    """Read flat x,y keypoints, allowing filename-only rows for empty roots."""
    if not path.is_file():
        raise ValueError(f"Missing Four Crops point annotations: {path}")
    rows: dict[PurePosixPath, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for line, row in enumerate(csv.reader(input_file), start=1):
            if not row:
                raise ValueError(f"Empty row in {path} line {line}.")
            image = _safe_image_name(row[0], path, line)
            coordinates = [value.strip() for value in row[1:]]
            if len(coordinates) % 2:
                raise ValueError(f"Expected x,y coordinate pairs in {path} line {line}.")
            try:
                values = [float(value) for value in coordinates]
            except ValueError as error:
                raise ValueError(f"Invalid coordinate in {path} line {line}.") from error
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Coordinates must be finite and nonnegative in {path} line {line}.")
            if image in rows:
                raise ValueError(f"Duplicate image in {path} line {line}: {image}")
            rows[image] = coordinates
    return rows


def _validate_pair(trl_file: Path, points_file: Path, image_root: Path) -> None:
    """Validate that TRL and point manifests describe the same images."""
    trl_images = {image for image, _ in _read_trl(trl_file, image_root)}
    point_images = set(_read_points(points_file))
    if trl_images != point_images:
        missing = sorted(str(name) for name in trl_images - point_images)
        extra = sorted(str(name) for name in point_images - trl_images)
        raise ValueError(
            f"Four Crops annotation image sets differ in {trl_file.parent}; "
            f"missing points={missing[:3]}, extra points={extra[:3]}."
        )


def resolve_four_crops_split(
    subset_dir: str | Path,
    subset: str,
    split: str,
    manifest_dir: str | Path | None = None,
) -> FourCropsSplit:
    """Resolve a published split, combining Dataset 4 acquisitions when needed."""
    root = Path(subset_dir).resolve()
    if subset not in {"dataset_1", "dataset_2", "dataset_3", "dataset_4"}:
        raise ValueError(f"Unknown Four Crops subset: {subset!r}")
    if split not in {"Train", "Val", "Test"}:
        raise ValueError(f"Unknown Four Crops split: {split!r}")
    if subset in {"dataset_3", "dataset_4"} and split != "Test":
        raise ValueError(f"Four Crops {subset} is inference-only and supports Test.")

    if subset in {"dataset_1", "dataset_2"}:
        image_root = root / f"sub_{split}"
        result = FourCropsSplit(
            image_root / f"{split}_TRL.csv",
            image_root / f"{split}_pointsOutput.csv",
            image_root,
        )
        _validate_pair(result.trl_file, result.points_file, result.base_dir)
        return result

    if subset == "dataset_3":
        result = FourCropsSplit(root / "TRL.csv", root / "pointsOutput.csv", root)
        _validate_pair(result.trl_file, result.points_file, result.base_dir)
        return result

    if manifest_dir is None:
        raise ValueError("A manifest directory is required for Four Crops dataset_4.")
    acquisitions = sorted(
        directory
        for directory in root.rglob("*")
        if directory.is_dir()
        and (directory / "TRL.csv").is_file()
        and (directory / "pointsOutput.csv").is_file()
    )
    if not acquisitions:
        raise ValueError(f"No Dataset 4 acquisition annotations were found in {root}.")

    output_dir = Path(manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trl_output = output_dir / "Test_TRL.csv"
    points_output = output_dir / "Test_pointsOutput.csv"
    combined_trl: list[list[str]] = []
    combined_points: list[list[str]] = []
    for acquisition in acquisitions:
        local_trl = _read_trl(acquisition / "TRL.csv", acquisition)
        local_points = _read_points(acquisition / "pointsOutput.csv")
        if {name for name, _ in local_trl} != set(local_points):
            raise ValueError(f"Four Crops annotation image sets differ in {acquisition}.")
        prefix = PurePosixPath(acquisition.relative_to(root).as_posix())
        for image, trl in local_trl:
            combined_name = str(prefix / image)
            combined_trl.append([combined_name, trl])
            combined_points.append([combined_name, *local_points[image]])
    with trl_output.open("w", encoding="utf-8", newline="") as output_file:
        csv.writer(output_file, lineterminator="\n").writerows(combined_trl)
    with points_output.open("w", encoding="utf-8", newline="") as output_file:
        csv.writer(output_file, lineterminator="\n").writerows(combined_points)
    result = FourCropsSplit(trl_output, points_output, root)
    _validate_pair(result.trl_file, result.points_file, result.base_dir)
    return result
