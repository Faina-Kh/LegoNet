"""Pure IoU matching for per-object evaluation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch

from legonet import config
from legonet.eval.detection_eval import compute_overlap


MatchResult = tuple[Any, Optional[int], float, bool]


def assign_detection_to_gt(
    detection: Any,
    annotations: Any,
    detected_annotations: Iterable[int],
    iou_threshold: float,
) -> tuple[Optional[int], float, bool]:
    """Match one detection to the highest-IoU unclaimed GT box."""
    if len(annotations) == 0:
        return None, -1.0, False

    overlaps = compute_overlap(
        np.expand_dims(np.asarray(detection), axis=0),
        np.asarray(annotations),
    )
    assigned_annotation = int(np.argmax(overlaps, axis=1)[0])
    max_overlap = float(overlaps[0, assigned_annotation])
    is_new_match = (
        max_overlap >= iou_threshold
        and assigned_annotation not in detected_annotations
    )
    return assigned_annotation, max_overlap, is_new_match


def match_detections_to_gt(
    detections: Iterable[Any],
    annotations: Any,
    iou_threshold: float,
) -> list[MatchResult]:
    """Greedily match score-ordered detections to each GT box at most once."""
    detected_annotations: list[int] = []
    matches: list[MatchResult] = []

    for detection in detections:
        assigned_annotation, max_overlap, is_new_match = assign_detection_to_gt(
            detection,
            annotations,
            detected_annotations,
            iou_threshold,
        )
        if is_new_match and assigned_annotation is not None:
            detected_annotations.append(assigned_annotation)
        matches.append(
            (detection, assigned_annotation, max_overlap, is_new_match)
        )

    return matches


def choose_boxes_by_iou_and_precision(
    detections: Any,
    annotations: Any,
    detection_scores: Any,
) -> list[Any]:
    """Assign score-ordered model detections to GT box identifiers.

    The name retains the historical ``precision`` terminology even though the
    active legacy behavior returns assignments immediately after IoU matching.
    """
    annotation_boxes = annotations[0, :, :4]
    score_order = np.argsort(-detection_scores.cpu())
    detections = detections[score_order, :]
    detected_annotations = []
    assignments = []

    for detection in detections:
        if annotation_boxes.shape[0] == 0:
            continue

        overlaps = compute_overlap(
            np.expand_dims(detection.cpu(), axis=0),
            annotation_boxes.cpu().numpy(),
        )
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]
        is_new_match = (
            max_overlap >= config.Detection.iou_threshold
            and assigned_annotation not in detected_annotations
        )
        if is_new_match:
            detected_annotations.append(assigned_annotation)
            gt_box_id = float(annotations[0][assigned_annotation[0]][5])
        else:
            gt_box_id = -1.0
        assignments.append(
            torch.cat(
                (
                    detection.unsqueeze(0),
                    torch.tensor([gt_box_id])
                    .unsqueeze(0)
                    .to(config.General.device),
                ),
                dim=-1,
            )
        )
    return assignments


# Compatibility aliases retained while callers migrate to the public names.
_assign_detection_to_gt = assign_detection_to_gt
_match_detections_to_gt = match_detections_to_gt
choose_boxes_by_IoUandPrc = choose_boxes_by_iou_and_precision
