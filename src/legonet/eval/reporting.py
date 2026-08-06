"""CSV artifact writers for LegoNet evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


COUNT_COLUMNS = (
    "img", "crop", "gt_count", "pred_count", "label", "score",
    "max_overlap",
)
ATTRIBUTE_COLUMNS = (
    "img", "crop", "gt_color", "pred_color", "label", "score",
    "max_overlap", "gt_TRL", "pred_TRL", "gt_dia", "pred_dia",
)
COUNT_SUMMARY_COLUMNS = (
    "img", "gt_count", "pred", "label", "score", "max_overlap",
)
ATTRIBUTE_SUMMARY_COLUMNS = (
    "img", "gt_color", "pred_color", "label", "score", "max_overlap",
    "gt_TRL", "pred_TRL", "gt_dia", "pred_dia",
)


def _write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Write one CSV with a header and prebuilt rows."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        writer.writerows(rows)


def _detection_rows(
    records: Mapping[str, Mapping[str, Sequence[Any]]],
    attributes: bool,
) -> list[list[Any]]:
    """Build rows for every predicted crop."""
    rows = []
    for image_name, image_records in records.items():
        for index in range(len(image_records["score"])):
            if attributes:
                row = [
                    image_name,
                    index,
                    image_records["color_gt"][index],
                    image_records["color_pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                    image_records["TRL_gt"][index],
                    image_records["TRL_pred"][index],
                    image_records["dia_gt"][index],
                    image_records["dia_pred"][index],
                ]
            else:
                row = [
                    image_name,
                    index,
                    image_records["gt_count"][index],
                    image_records["pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                ]
            rows.append(row)
    return rows


def _summary_rows(
    records: Mapping[str, Mapping[str, Sequence[Any]]],
    attributes: bool,
    empty_image_placeholders: bool,
) -> list[list[Any]]:
    """Build unmatched-GT or no-prediction rows using legacy placeholders."""
    rows = []
    prediction_key = "TRL_pred" if attributes else "pred"
    for image_name, image_records in records.items():
        predictions = image_records[prediction_key] if image_records else ()
        for index in range(len(predictions)):
            if attributes:
                row = [
                    image_name,
                    image_records["color_gt"][index],
                    image_records["color_pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                    image_records["TRL_gt"][index],
                    image_records["TRL_pred"][index],
                    image_records["dia_gt"][index],
                    image_records["dia_pred"][index],
                ]
            else:
                row = [
                    image_name,
                    image_records["gt_count"][index],
                    image_records["pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                ]
            rows.append(row)

        if predictions:
            continue
        if empty_image_placeholders:
            rows.append(
                [image_name, -1, -1, -1, -1, -1, 0, 0, 0, 0]
                if attributes
                else [image_name, 0, 0, -1, -1, -1]
            )
        elif not image_records:
            rows.append([image_name, "None"])
    return rows


def write_evaluation_artifacts(
    output_directory: str,
    state: Mapping[str, Any],
    attributes: bool,
) -> None:
    """Write the three historical per-object evaluation CSV artifacts."""
    output_path = Path(output_directory)
    _write_rows(
        output_path / "detections_data_any_crop.csv",
        ATTRIBUTE_COLUMNS if attributes else COUNT_COLUMNS,
        _detection_rows(state["detections_data_any_crop"], attributes),
    )
    summary_columns = (
        ATTRIBUTE_SUMMARY_COLUMNS if attributes else COUNT_SUMMARY_COLUMNS
    )
    _write_rows(
        output_path / "not_found_gt_count.csv",
        summary_columns,
        _summary_rows(state["not_found_gt"], attributes, False),
    )
    _write_rows(
        output_path / "images_without_detections.csv",
        summary_columns,
        _summary_rows(state["no_predictions"], attributes, True),
    )


def write_keypoint_precision_recall(
    output_directory: str,
    recall: Sequence[Any],
    precision: Sequence[Any],
) -> None:
    """Write the historical keypoint precision-recall CSV artifact."""
    if len(recall) != len(precision):
        raise ValueError("Recall and precision arrays must have equal length.")
    _write_rows(
        Path(output_directory) / "parts_recall_precision.csv",
        ("recall", "precision"),
        list(zip(recall, precision)),
    )
