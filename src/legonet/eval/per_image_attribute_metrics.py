"""Dataset metrics calculated from one aggregate value per image."""

from dataclasses import dataclass
from typing import Sequence

from legonet.eval.per_object_result import RegressionMetrics
from legonet.eval.regression_metrics import compute_regression_metrics


@dataclass(frozen=True)
class PerImageAttributeMetrics:
    """Metrics across image-level TRL, diameter, and color aggregates."""

    trl: RegressionMetrics
    diameter: RegressionMetrics
    color: RegressionMetrics


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
