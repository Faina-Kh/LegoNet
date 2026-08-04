"""Per-image metrics derived from IoU-matched per-object records."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from legonet.eval.classification_metrics import compute_classification_metrics
from legonet.eval.per_object_result import (
    ClassificationMetrics,
    ClassificationType,
    RegressionMetrics,
)
from legonet.eval.regression_metrics import compute_regression_metrics


@dataclass(frozen=True)
class RootsPerImageMetrics:
    """Matched-object attribute metrics for one roots image."""

    trl: RegressionMetrics
    diameter: RegressionMetrics
    color: ClassificationMetrics


def _valid_pairs(
    ground_truth: Sequence[Any], predictions: Sequence[Any]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return aligned numeric pairs whose GT value is not the legacy sentinel."""
    if len(ground_truth) != len(predictions):
        raise ValueError("Per-image ground truth and predictions must align.")
    pairs = [
        (float(target), float(prediction))
        for target, prediction in zip(ground_truth, predictions)
        if float(target) != -1.0
    ]
    if not pairs:
        return (), ()
    targets, estimates = zip(*pairs)
    return tuple(targets), tuple(estimates)


def compute_roots_per_image_metrics(
    image_records: Mapping[str, Sequence[Any]],
) -> RootsPerImageMetrics:
    """Compute metrics from one image's matched roots attribute records."""
    trl_targets, trl_predictions = _valid_pairs(
        image_records["TRL_gt"], image_records["TRL_pred"]
    )
    diameter_targets, diameter_predictions = _valid_pairs(
        image_records["dia_gt"], image_records["dia_pred"]
    )
    color_targets, color_predictions = _valid_pairs(
        image_records["color_gt"], image_records["color_pred"]
    )
    return RootsPerImageMetrics(
        trl=compute_regression_metrics(trl_targets, trl_predictions),
        diameter=compute_regression_metrics(
            diameter_targets, diameter_predictions
        ),
        color=compute_classification_metrics(
            tuple(int(value) for value in color_targets),
            tuple(int(round(value)) for value in color_predictions),
            class_labels=(0, 1),
            class_names=("non_white", "white"),
            classification_type=ClassificationType.BINARY,
        ),
    )
