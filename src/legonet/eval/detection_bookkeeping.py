"""State updates derived from per-image detection-to-GT matches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from legonet.eval.matching import match_detections_to_gt


@dataclass(frozen=True)
class DetectionBookkeepingResult:
    """Aligned GT values and overlaps produced by detection bookkeeping."""

    count_ground_truth: list[Any]
    length_ground_truth: list[Any]
    diameter_ground_truth: list[Any]
    color_ground_truth: list[Any]
    max_overlaps: Optional[list[float]]


def _append_false_positive(
    state: dict[str, Any],
    image_name: str,
    attributes: bool,
    count_ground_truth: list[Any],
    length_ground_truth: list[Any],
    diameter_ground_truth: list[Any],
    color_ground_truth: list[Any],
    score: float,
    overlap: float,
) -> None:
    """Record one prediction that did not claim an annotated GT box."""
    record = state["detections_data_any_crop"][image_name]
    state["FP"] += 1
    record["score"].append(score)
    record["max_overlap"].append(overlap)
    record["label"].append(0)
    record["gt_box_id"].append(torch.tensor(-1, dtype=float))

    if not attributes:
        count_ground_truth.append(-1)
        record["gt_count"].append(-1)
        return

    length_ground_truth.append(-1)
    diameter_ground_truth.append(-1)
    color_ground_truth.append(-1)
    record["length_gt"].append(-1)
    record["dia_gt"].append(-1)
    record["color_gt"].append(-1)


def record_detection_bookkeeping(
    state: dict[str, Any],
    image_name: str,
    predicted_boxes: Sequence[Any],
    annotated_boxes: Any,
    gt_values: Sequence[Any],
    attributes: bool,
    counting: bool,
    iou_threshold: float,
) -> DetectionBookkeepingResult:
    """Match detections and update found, false-positive, and missed-GT state."""
    count_ground_truth: list[Any] = []
    length_ground_truth: list[Any] = []
    diameter_ground_truth: list[Any] = []
    color_ground_truth: list[Any] = []
    matched_gt_value_indices: list[int] = []
    max_overlaps: Optional[list[float]] = []

    if predicted_boxes and len(annotated_boxes) > 0:
        boxes = torch.cat(
            [torch.tensor(box).unsqueeze(dim=0) for box in predicted_boxes],
            dim=0,
        )
        scores = np.asarray([box[4] for box in predicted_boxes], dtype=float)
        score_order = np.argsort(-scores)
        sorted_matches = match_detections_to_gt(
            boxes[score_order], annotated_boxes, iou_threshold
        )
        matches = [None] * len(sorted_matches)
        for sorted_index, original_index in enumerate(score_order):
            matches[original_index] = sorted_matches[sorted_index]

        record = state["detections_data_any_crop"][image_name]
        for detection, assigned_annotation, overlap, is_new_match in matches:
            record["score"].append(float(detection[4]))
            record["max_overlap"].append(overlap)
            max_overlaps.append(overlap)

            if not is_new_match or assigned_annotation is None:
                # Score and overlap were already stored above.
                record["score"].pop()
                record["max_overlap"].pop()
                _append_false_positive(
                    state,
                    image_name,
                    attributes,
                    count_ground_truth,
                    length_ground_truth,
                    diameter_ground_truth,
                    color_ground_truth,
                    float(detection[4]),
                    overlap,
                )
                continue

            state["found_orig_objects"] += 1
            record["label"].append(1)
            gt_box_id = annotated_boxes[assigned_annotation][5]
            record["gt_box_id"].append(gt_box_id)

            matching_value_index = next(
                (
                    index
                    for index, values in enumerate(gt_values)
                    if values[2] == gt_box_id
                ),
                None,
            )
            if matching_value_index is None:
                if attributes:
                    length_ground_truth.append(-1)
                    diameter_ground_truth.append(-1)
                    color_ground_truth.append(-1)
                    record["length_gt"].append(0)
                    record["dia_gt"].append(0)
                    record["color_gt"].append(-1)
                else:
                    count_ground_truth.append(-1)
                    record["gt_count"].append(0)
                continue

            values = gt_values[matching_value_index]
            matched_gt_value_indices.append(matching_value_index)
            if attributes:
                color, length, diameter = values[0], values[3], values[4]
                color_ground_truth.append(color)
                length_ground_truth.append(length)
                diameter_ground_truth.append(diameter)
                record["color_gt"].append(color)
                record["length_gt"].append(length)
                record["dia_gt"].append(diameter)
            else:
                count_ground_truth.append(values[0])
                record["gt_count"].append(values[0])

    elif predicted_boxes:
        max_overlaps = None
        for detection in predicted_boxes:
            _append_false_positive(
                state,
                image_name,
                attributes,
                count_ground_truth,
                length_ground_truth,
                diameter_ground_truth,
                color_ground_truth,
                float(detection[4]),
                -1,
            )

    unmatched_indices = list(
        set(np.arange(len(gt_values))) - set(matched_gt_value_indices)
    )
    missed_record = state["not_found_gt"][image_name]
    for index in unmatched_indices:
        values = gt_values[index]
        missed_record["label"].append(1)
        missed_record["score"].append(-1)
        missed_record["max_overlap"].append(-1)
        if counting:
            missed_record["gt_count"].append(values[0])
            missed_record["pred"].append(-1)
        if attributes:
            missed_record["length_gt"].append(values[3])
            missed_record["length_pred"].append(-1)
            missed_record["dia_gt"].append(values[4])
            missed_record["dia_pred"].append(-1)
            missed_record["color_gt"].append(values[0])
            missed_record["color_pred"].append(-1)

    return DetectionBookkeepingResult(
        count_ground_truth=count_ground_truth,
        length_ground_truth=length_ground_truth,
        diameter_ground_truth=diameter_ground_truth,
        color_ground_truth=color_ground_truth,
        max_overlaps=max_overlaps,
    )
