"""Pure metric calculations for continuous-valued attributes."""

from typing import Optional, Sequence

import numpy as np

from legonet.eval.per_object_result import RegressionMetrics


def compute_regression_metrics(
    ground_truth: Sequence[float],
    predictions: Sequence[float],
    relative_errors: Optional[Sequence[float]] = None,
    preserve_zero_variance_division: bool = False,
) -> RegressionMetrics:
    """Calculate publication regression metrics for aligned observations.

    Callers may supply task-specific relative errors when zero-valued targets
    require an established inclusion policy. Otherwise, relative error is
    averaged over nonzero ground-truth values. A constant ground-truth target
    preserves the counting evaluator's undefined 1-FVU result as ``NaN``.
    ``preserve_zero_variance_division`` retains the roots evaluator's older
    NumPy division behavior (``NaN`` or ``-inf``) during migration.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length.")
    if relative_errors is not None and len(relative_errors) > len(ground_truth):
        raise ValueError(
            "Relative errors cannot outnumber aligned observations."
        )

    sample_count = len(ground_truth)
    if sample_count == 0:
        return RegressionMetrics(eligible_samples=0, evaluated_samples=0)

    targets = np.asarray(ground_truth, dtype=float)
    estimates = np.asarray(predictions, dtype=float)
    errors = targets - estimates
    mean_absolute_error = float(np.mean(np.abs(errors)))
    mean_squared_error = float(np.mean(errors ** 2))

    # ``relative error`` in the code is the per-sample quantity averaged to
    # obtain MRD (Mean Relative Deviation), as defined in the LegoNet papers.
    if relative_errors is None:
        nonzero_targets = targets != 0
        relative_error_values = np.abs(
            errors[nonzero_targets] / targets[nonzero_targets]
        )
    else:
        relative_error_values = np.asarray(relative_errors, dtype=float)
    mean_relative_error = (
        float(np.mean(relative_error_values))
        if len(relative_error_values) > 0
        else None
    )

    target_variance = float(np.var(targets))
    if target_variance > 0:
        one_minus_fvu = 1.0 - mean_squared_error / target_variance
    elif preserve_zero_variance_division:
        with np.errstate(divide="ignore", invalid="ignore"):
            one_minus_fvu = float(
                1.0 - np.divide(mean_squared_error, target_variance)
            )
    else:
        one_minus_fvu = float("nan")
    return RegressionMetrics(
        eligible_samples=sample_count,
        evaluated_samples=sample_count,
        mean_absolute_error=mean_absolute_error,
        mean_squared_error=mean_squared_error,
        mean_relative_error=mean_relative_error,
        one_minus_fvu=one_minus_fvu,
    )
