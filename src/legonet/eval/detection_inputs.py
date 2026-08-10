"""Preparation of detection-evaluation inputs for one image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

import numpy as np


@dataclass(frozen=True)
class DetectionEvaluationInputs:
    """Predictions and GT data consumed by detection evaluation and drawing."""

    detections: np.ndarray
    point_annotations: list[Any]
    ground_truth_boxes: Any
    original_scale_annotations: np.ndarray


def prepare_detection_evaluation_inputs(
    state: MutableMapping[str, Any],
    dataset: Any,
    data,
    image_name: str,
    predicted_boxes: np.ndarray,
    object_scores: np.ndarray,
    scale,
    *,
    have_gt: bool,
    has_ground_truth_boxes: bool,
) -> DetectionEvaluationInputs:
    """Build detection rows and original-image GT annotations for one image."""
    detections = np.concatenate(
        (predicted_boxes, np.asarray([object_scores]).T), axis=1
    )
    state["all_detections"].append(detections)

    point_annotations = []
    ground_truth_boxes = []
    original_scale_annotations = np.array([])
    if have_gt:
        point_annotations = dataset.image_data_points_location[image_name]
        if len(point_annotations) == 0:
            dataset.image_data_points_location[image_name] = []
            point_annotations = dataset.image_data_points_location[image_name]

        current_annotations = []
        if has_ground_truth_boxes:
            ground_truth_boxes = data["bbox_annot"][0]
            for box in ground_truth_boxes:
                coordinates = [
                    box[0].numpy() / scale,
                    box[1].numpy() / scale,
                    box[2].numpy() / scale,
                    box[3].numpy() / scale,
                ]
                current_annotations.append(np.asarray(coordinates).reshape(4))
        original_scale_annotations = np.asarray(current_annotations)
        state["all_annotations"].append(original_scale_annotations)

    return DetectionEvaluationInputs(
        detections,
        point_annotations,
        ground_truth_boxes,
        original_scale_annotations,
    )
