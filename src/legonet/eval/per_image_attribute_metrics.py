"""Dataset metrics calculated from one aggregate value per image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from legonet.eval.per_object_result import RegressionMetrics
from legonet.eval.regression_metrics import compute_regression_metrics


@dataclass(frozen=True)
class PerImageAttributeMetrics:
    """Metrics across image-level TRL, diameter, and color aggregates."""

    trl: RegressionMetrics
    diameter: RegressionMetrics
    color: RegressionMetrics


@dataclass(frozen=True)
class ImageAttributeAggregates:
    """GT and prediction aggregates for one image's matched objects."""

    trl_ground_truth: float
    trl_prediction: float
    diameter_ground_truth: float
    diameter_prediction: float
    color_ground_truth: float
    color_prediction: float


def aggregate_matched_image_attributes(
    trl_ground_truth: Sequence[float],
    trl_predictions: Sequence[float],
    diameter_ground_truth: Sequence[float],
    diameter_predictions: Sequence[float],
    color_ground_truth: Sequence[float],
    color_predictions: Sequence[float],
) -> Optional[ImageAttributeAggregates]:
    """Aggregate aligned values for IoU-matched objects in one image."""
    lengths = {
        len(trl_ground_truth),
        len(trl_predictions),
        len(diameter_ground_truth),
        len(diameter_predictions),
        len(color_ground_truth),
        len(color_predictions),
    }
    if len(lengths) != 1:
        raise ValueError("Matched attribute arrays must have equal length.")
    matched_indices = [
        index for index, target in enumerate(trl_ground_truth) if target != -1
    ]
    if not matched_indices:
        return None

    def selected(values: Sequence[float]) -> list[float]:
        return [float(values[index]) for index in matched_indices]

    matched_trl_gt = selected(trl_ground_truth)
    matched_trl_pred = selected(trl_predictions)
    matched_diameter_gt = selected(diameter_ground_truth)
    matched_diameter_pred = selected(diameter_predictions)
    matched_color_gt = selected(color_ground_truth)
    matched_color_pred = selected(color_predictions)
    return ImageAttributeAggregates(
        trl_ground_truth=sum(matched_trl_gt),
        trl_prediction=sum(matched_trl_pred),
        diameter_ground_truth=sum(matched_diameter_gt) / len(matched_indices),
        diameter_prediction=sum(matched_diameter_pred) / len(matched_indices),
        color_ground_truth=sum(matched_color_gt) / len(matched_indices),
        color_prediction=sum(matched_color_pred) / len(matched_indices),
    )


def compute_per_image_attribute_metrics(
    trl_ground_truth: Sequence[float],
    trl_predictions: Sequence[float],
    diameter_ground_truth: Sequence[float],
    diameter_predictions: Sequence[float],
    color_ground_truth: Sequence[float],
    color_predictions: Sequence[float],
) -> PerImageAttributeMetrics:
    """Calculate metrics across corresponding per-image aggregate values.

    Each TRL value is an image-level sum. Diameter and color values are
    image-level means of their respective per-object values.
    """
    return PerImageAttributeMetrics(
        trl=compute_regression_metrics(trl_ground_truth, trl_predictions),
        diameter=compute_regression_metrics(
            diameter_ground_truth, diameter_predictions
        ),
        color=compute_regression_metrics(color_ground_truth, color_predictions),
    )
