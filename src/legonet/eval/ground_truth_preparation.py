"""Ground-truth preparation for one per-object evaluation image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

import numpy as np
import torch

from legonet.eval.evaluation_policy import EvaluationTask, should_include_image


@dataclass(frozen=True)
class PreparedGroundTruth:
    """Normalized GT records and control-flow decision for one image."""

    all_boxes: Any
    annotated_boxes: Any
    matched_counts: list[Any]
    include_image: bool
    skip_reason: str | None
    image_gt_average: float
    image_trl_sum: float
    image_diameter_average: float


def split_boxes_by_annotations(box_annotations, count_annotations):
    """Split all GT boxes from boxes carrying count/attribute annotations."""
    all_boxes = []
    boxes_with_annotations = []
    matched_counts = []
    for box in box_annotations:
        box_id = box[5]
        box_tensor = torch.tensor(box).unsqueeze(dim=0)
        all_boxes.append(box_tensor)
        for count_annotation in count_annotations:
            if count_annotation[2] == box_id:
                matched_counts.append(count_annotation)
                boxes_with_annotations.append(box_tensor)
                break
    return all_boxes, boxes_with_annotations, matched_counts


def prepare_image_ground_truth(
    state: MutableMapping[str, Any], image_name: str, box_annotations,
    count_annotations, *, have_gt: bool, attributes: bool,
) -> PreparedGroundTruth:
    """Record per-image GT statistics and normalize eligible object boxes."""
    image_gt_average, image_trl_sum, image_diameter_average = _record_statistics(
        state, image_name, count_annotations, attributes=attributes
    )
    if not have_gt:
        return PreparedGroundTruth([], [], [], True, None, image_gt_average,
                                   image_trl_sum, image_diameter_average)

    all_boxes, annotated_boxes, matched_counts = split_boxes_by_annotations(
        box_annotations, count_annotations
    )
    state["num_of_gt_boxes"] += len(all_boxes)
    include_image = should_include_image(
        EvaluationTask.PER_OBJECT, all_boxes, annotated_boxes
    )
    if not include_image:
        reason = "no_gt_boxes" if not all_boxes else "no_annotated_boxes"
        return PreparedGroundTruth(all_boxes, annotated_boxes, matched_counts,
                                   False, reason, image_gt_average,
                                   image_trl_sum, image_diameter_average)

    for count in matched_counts:
        state["all_data_gt_count"].append(count[0])
        state["gt_objects_withGTpoints"] += 1
        if attributes:
            state.setdefault("all_data_gt_TRL", []).append(count[3])
            state.setdefault("all_data_gt_dia", []).append(count[4])
            state.setdefault("all_data_gt_color", []).append(count[0])

    return PreparedGroundTruth(
        _concatenate_boxes(all_boxes), _concatenate_boxes(annotated_boxes),
        matched_counts, True, None, image_gt_average, image_trl_sum,
        image_diameter_average,
    )


def _record_statistics(state, image_name, count_annotations, *, attributes):
    if len(count_annotations) > 0:
        image_gt_average = float(np.sum(count_annotations[:, 0]) / count_annotations.shape[0])
        image_trl_sum = float(np.sum(count_annotations[:, 3])) if attributes else 0.0
        image_diameter_average = (float(np.sum(count_annotations[:, 4]) /
                                        count_annotations.shape[0])
                                  if attributes else 0.0)
    else:
        image_gt_average = image_trl_sum = image_diameter_average = 0.0

    if len(count_annotations) > 0 or attributes:
        state["per_im_gt_avg"].append(image_gt_average)
        state["per_im_gt_avg_dict"][image_name] = image_gt_average
        if attributes:
            state.setdefault("TRL_per_im_gt_sum", []).append(image_trl_sum)
            state.setdefault("TRL_per_im_gt_sum_dict", {})[image_name] = image_trl_sum
            state.setdefault("dia_per_im_gt_avg", []).append(image_diameter_average)
            state.setdefault("dia_per_im_gt_avg_dict", {})[image_name] = image_diameter_average
    return image_gt_average, image_trl_sum, image_diameter_average


def _concatenate_boxes(boxes):
    if len(boxes) > 1:
        return torch.cat(boxes, dim=0)
    if len(boxes) == 1:
        return boxes[0]
    return boxes
