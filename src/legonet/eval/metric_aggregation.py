"""Post-loop metric aggregation for per-object evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from legonet.eval.classification_metrics import compute_classification_metrics
from legonet.eval.per_object_result import ClassificationType
from legonet.eval.regression_metrics import compute_regression_metrics


@dataclass(frozen=True)
class PostLoopMetrics:
    """Metrics calculated after all evaluation images have been processed."""

    had_predictions: bool
    has_original_gt: bool
    precision_detection: float
    crop_count_metrics: dict[str, float | int] | None = None
    count_mae: float = -1
    count_relative_error: float = -1
    count_mse: float = -1
    count_agreement: float = -1
    count_fvu: float = -1
    trl_mae: float = -1
    trl_relative_error: float = 100000
    trl_mse: float = -1
    trl_fvu: float = -1
    diameter_mae: float = -1
    diameter_relative_error: float = -1
    diameter_mse: float = -1
    diameter_fvu: float = -1
    color_metrics: Any = None


def compute_positive_crop_count_metrics(
    ground_truth_counts: Iterable[float],
    predicted_counts: Iterable[float],
) -> dict[str, float | int]:
    """Compute counting diagnostics for crops containing annotated points."""
    ground_truth = np.asarray(list(ground_truth_counts), dtype=float)
    predictions = np.asarray(list(predicted_counts), dtype=float)
    if ground_truth.shape != predictions.shape:
        raise ValueError("Crop ground-truth and prediction counts must align")

    positive_mask = ground_truth > 0
    positive_ground_truth = ground_truth[positive_mask]
    positive_predictions = predictions[positive_mask]
    num_positive = int(np.count_nonzero(positive_mask))
    metrics: dict[str, float | int] = {
        "num_total": int(ground_truth.size),
        "num_positive": num_positive,
        "num_empty": int(np.count_nonzero(ground_truth == 0)),
        "mae": -1.0,
        "mse": -1.0,
        "mean_relative_error": -1.0,
        "exact_agreement": -1.0,
    }
    if num_positive == 0:
        return metrics

    absolute_errors = np.abs(positive_ground_truth - positive_predictions)
    metrics.update(
        {
            "mae": float(np.mean(absolute_errors)),
            "mse": float(np.mean(absolute_errors**2)),
            "mean_relative_error": float(np.mean(absolute_errors / positive_ground_truth)),
            "exact_agreement": float(np.mean(positive_ground_truth == positive_predictions)),
        }
    )
    return metrics


def aggregate_post_loop_metrics(
    state: Mapping[str, Any],
    *,
    attributes: bool,
) -> PostLoopMetrics:
    """Flatten per-image records and calculate final evaluation metrics."""
    if attributes:
        return _aggregate_attribute_metrics(state)
    return _aggregate_counting_metrics(state)


def _aggregate_counting_metrics(state: Mapping[str, Any]) -> PostLoopMetrics:
    crop_gt = []
    crop_predictions = []
    original_gt = []
    original_predictions = []
    for image_index, image_crop_gt in enumerate(state["all_crops_GT_counts"]):
        for crop_index, gt_count in enumerate(image_crop_gt):
            crop_gt.append(gt_count)
            crop_predictions.append(state["all_predicted_counts"][image_index][crop_index])
            image_original_gt = state["all_orig_GT_counts"][image_index]
            if image_original_gt and image_original_gt[crop_index] != -1:
                original_gt.append(image_original_gt[crop_index])
                original_predictions.append(state["all_predicted_counts"][image_index][crop_index])

    crop_metrics = compute_positive_crop_count_metrics(crop_gt, crop_predictions)
    found = state["found_orig_objects"]
    detections = found + state["FP"]
    precision = found / detections if detections > 0 else -1
    if not original_gt:
        return PostLoopMetrics(
            had_predictions=bool(state["all_predicted_counts"]),
            has_original_gt=False,
            precision_detection=precision,
            crop_count_metrics=crop_metrics,
        )

    metrics = compute_regression_metrics(
        original_gt,
        original_predictions,
        relative_errors=state["orig_rel_error"],
    )
    return PostLoopMetrics(
        had_predictions=bool(state["all_predicted_counts"]),
        has_original_gt=True,
        precision_detection=precision,
        crop_count_metrics=crop_metrics,
        count_mae=metrics.mean_absolute_error,
        count_relative_error=(
            metrics.mean_relative_error if metrics.mean_relative_error is not None else -1
        ),
        count_mse=metrics.mean_squared_error,
        count_agreement=float(np.mean(np.asarray(original_gt) == np.asarray(original_predictions))),
        count_fvu=1 - metrics.one_minus_fvu,
    )


def _aggregate_attribute_metrics(state: Mapping[str, Any]) -> PostLoopMetrics:
    trl_gt = []
    trl_predictions = []
    diameter_gt = []
    diameter_predictions = []
    color_gt = []
    color_predictions = []
    for image_index, image_predictions in enumerate(state["all_predicted_TRL"]):
        for object_index, trl_prediction in enumerate(image_predictions):
            image_trl_gt = state["all_orig_GT_TRL"][image_index]
            if image_trl_gt and image_trl_gt[object_index] != -1:
                trl_gt.append(image_trl_gt[object_index])
                trl_predictions.append(trl_prediction)
                diameter_gt.append(state["all_orig_GT_dia"][image_index][object_index])
                diameter_predictions.append(state["all_predicted_dia"][image_index][object_index])
            if state["all_orig_GT_color"][image_index][object_index] != -1:
                color_gt.append(int(state["all_orig_GT_color"][image_index][object_index]))
                color_predictions.append(int(state["all_predicted_color"][image_index][object_index]))

    found = state["found_orig_objects"]
    detections = found + state["FP"]
    precision = found / detections if detections > 0 else -1
    if not trl_gt:
        return PostLoopMetrics(
            had_predictions=bool(state["all_predicted_TRL"]),
            has_original_gt=False,
            precision_detection=precision,
        )

    trl_metrics = compute_regression_metrics(
        trl_gt,
        trl_predictions,
        relative_errors=state["orig_rel_error_TRL"],
        preserve_zero_variance_division=True,
    )
    diameter_metrics = compute_regression_metrics(
        diameter_gt,
        diameter_predictions,
        relative_errors=state["orig_rel_error_dia"],
        preserve_zero_variance_division=True,
    )
    eligible_color_samples = sum(int(color) != -1 for color in state["all_data_gt_color"])
    color_metrics = compute_classification_metrics(
        color_gt,
        color_predictions,
        class_labels=(0, 1),
        class_names=("non_white", "white"),
        classification_type=ClassificationType.BINARY,
        eligible_samples=eligible_color_samples,
    )
    return PostLoopMetrics(
        had_predictions=bool(state["all_predicted_TRL"]),
        has_original_gt=True,
        precision_detection=precision,
        trl_mae=trl_metrics.mean_absolute_error,
        trl_relative_error=trl_metrics.mean_relative_error,
        trl_mse=trl_metrics.mean_squared_error,
        trl_fvu=1 - trl_metrics.one_minus_fvu,
        diameter_mae=diameter_metrics.mean_absolute_error,
        diameter_relative_error=diameter_metrics.mean_relative_error,
        diameter_mse=diameter_metrics.mean_squared_error,
        diameter_fvu=1 - diameter_metrics.one_minus_fvu,
        color_metrics=color_metrics,
    )
