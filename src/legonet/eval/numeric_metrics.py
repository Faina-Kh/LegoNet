"""Small task-neutral numerical helpers retained for legacy evaluators."""

from __future__ import annotations

from collections.abc import Sequence


def sum_of_differences(
    ground_truth: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Return the signed sum of aligned ground-truth/prediction differences."""
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length.")
    return float(sum(gt - prediction for gt, prediction in zip(ground_truth, predictions)))


def sum_of_absolute_differences(
    ground_truth: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Return the absolute sum of aligned ground-truth/prediction differences."""
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length.")
    return float(
        sum(abs(gt - prediction) for gt, prediction in zip(ground_truth, predictions))
    )
