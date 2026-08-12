"""Decode and align predictions produced for each detected object crop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, Sequence

import numpy as np
import torch

from legonet.eval.classification_metrics import decode_class_predictions
from legonet.eval.per_object_result import ClassificationType


@dataclass
class PreparedCropPredictions:
    """Aligned crop predictions and optional keypoint maps for one image."""

    sample_annotations: Any
    adjusted_original_boxes: list[Any]
    ground_truth_counts: list[Any]
    count_predictions: list[Any]
    trl_predictions: list[float]
    diameter_predictions: list[float]
    color_predictions: list[int]
    predicted_detection_maps: Any
    predicted_maps_to_draw: list[Any]
    ground_truth_detection_maps: Any


def prepare_crop_predictions(
    *,
    state: MutableMapping[str, Any],
    dataset: Any,
    image_name: str,
    predicted_boxes: Any,
    original_crop_boxes: Sequence[Any],
    estimation_outputs: Any,
    sample_annotations: Any,
    scale: Any,
    evaluates_attributes: bool,
    have_ground_truth: bool,
    estimate_type: str,
    crop_size: Sequence[int],
    point_center_map_builder: Callable[..., np.ndarray],
) -> PreparedCropPredictions:
    """Decode crop outputs and prepare aligned count/keypoint evaluation data."""
    adjusted_original_boxes: list[Any] = []
    ground_truth_counts: list[Any] = []
    count_predictions: list[Any] = []
    trl_predictions: list[float] = []
    diameter_predictions: list[float] = []
    color_predictions: list[int] = []
    predicted_detection_maps: Any = []
    predicted_maps_to_draw: list[Any] = []
    ground_truth_detection_maps: Any = []

    uses_keypoints = estimate_type == "withKeyPoints"
    original_detection_maps: list[Any] = []
    if uses_keypoints and estimation_outputs is not None:
        original_detection_maps = [
            estimation_outputs[1],
            estimation_outputs[2],
            estimation_outputs[3],
            estimation_outputs[4],
            estimation_outputs[6],
        ]
    if sample_annotations is not None:
        if "points_annot" not in sample_annotations:
            sample_annotations["points_annot"] = []

    if not isinstance(predicted_boxes, list):
        for crop_index in range(predicted_boxes.shape[0]):
            if estimation_outputs is not None:
                if evaluates_attributes:
                    attribute_output = estimation_outputs[0][crop_index]
                    decoded_color = int(
                        decode_class_predictions(
                            [attribute_output[0].cpu().item()],
                            ClassificationType.BINARY,
                        )[0]
                    )
                    trl_prediction = attribute_output[1].cpu().item()
                    diameter_prediction = attribute_output[2].cpu().item()
                    record = state["detections_data_any_crop"][image_name]
                    record["color_pred"].append(decoded_color)
                    record["TRL_pred"].append(trl_prediction)
                    record["dia_pred"].append(diameter_prediction)
                    color_predictions.append(decoded_color)
                    trl_predictions.append(trl_prediction)
                    diameter_predictions.append(diameter_prediction)
                else:
                    prediction = np.round(
                        estimation_outputs[0][crop_index].cpu().item()
                    )
                    state["detections_data_any_crop"][image_name]["pred"].append(
                        prediction
                    )
                    count_predictions.append(prediction)

            if len(original_crop_boxes) > 0:
                adjusted_original_boxes.append(original_crop_boxes[crop_index])

            if uses_keypoints and estimation_outputs is not None:
                predicted_detection_maps.append(
                    original_detection_maps[-1][crop_index]
                )
                predicted_maps_to_draw.append(
                    [
                        original_detection_maps[0][crop_index],
                        original_detection_maps[1][crop_index],
                        original_detection_maps[2][crop_index],
                        original_detection_maps[3][crop_index],
                        original_detection_maps[4][crop_index],
                    ]
                )
                if sample_annotations is not None and have_ground_truth:
                    center_map = point_center_map_builder(
                        dataset.image_data_points_location[image_name],
                        original_crop_boxes[crop_index],
                        scale,
                        predicted_detection_maps[-1].shape,
                        crop_size,
                    )
                    ground_truth_detection_maps.append(torch.tensor(center_map))

            if sample_annotations is not None and have_ground_truth:
                current_count = sample_annotations["points_annot"][0][
                    crop_index
                ].item()
                if current_count == 0:
                    state["crops_without_gt_points"] += 1
                ground_truth_counts.append(current_count)

    if sample_annotations is not None:
        maximum_crop_count = (
            np.max(ground_truth_counts) if ground_truth_counts else -1
        )
        if maximum_crop_count == -1 and not evaluates_attributes:
            sample_annotations = None
        elif uses_keypoints:
            if have_ground_truth:
                ground_truth_detection_maps = torch.cat(
                    [
                        torch.unsqueeze(point_map, dim=0)
                        for point_map in ground_truth_detection_maps
                    ],
                    dim=0,
                )
            if len(predicted_detection_maps) > 1:
                predicted_detection_maps = torch.cat(
                    [
                        torch.unsqueeze(point_map, dim=0)
                        for point_map in predicted_detection_maps
                    ],
                    dim=0,
                )
            else:
                predicted_detection_maps = predicted_detection_maps[0].unsqueeze(
                    dim=0
                )

    return PreparedCropPredictions(
        sample_annotations=sample_annotations,
        adjusted_original_boxes=adjusted_original_boxes,
        ground_truth_counts=ground_truth_counts,
        count_predictions=count_predictions,
        trl_predictions=trl_predictions,
        diameter_predictions=diameter_predictions,
        color_predictions=color_predictions,
        predicted_detection_maps=predicted_detection_maps,
        predicted_maps_to_draw=predicted_maps_to_draw,
        ground_truth_detection_maps=ground_truth_detection_maps,
    )
