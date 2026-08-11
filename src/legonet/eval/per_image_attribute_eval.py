"""Canonical evaluation boundary for per-image attribute models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from legonet.eval import attribute_estimation_eval as _legacy_evaluator


@dataclass(frozen=True)
class PerImageCheckpointMetrics:
    """Named scalar and optimization direction for per-image checkpoints."""

    metric_name: str
    metric_value: Optional[float]


def evaluate(
    dataloader: Any,
    dataset: Any,
    model: Any,
    args: Any,
    *,
    do_profile: bool = False,
) -> Optional[float]:
    """Evaluate a per-image model through the compatibility implementation."""
    return _legacy_evaluator.eval(
        dataloader,
        dataset,
        model,
        args,
        do_profile=do_profile,
    )


def evaluate_checkpoint_metrics(
    dataloader: Any,
    dataset: Any,
    model: Any,
    args: Any,
) -> PerImageCheckpointMetrics:
    """Evaluate and name the scalar used for per-image checkpoint selection."""
    value = evaluate(dataloader, dataset, model, args)
    metric_value = None if value is None else float(value)
    if getattr(model.estimator, "binary_model", False):
        return PerImageCheckpointMetrics(
            metric_name="classification_error_rate",
            metric_value=metric_value,
        )
    if args.network_type in {
        "per_image_estimation_keypoints",
        "per_image_estimation_regression",
    }:
        return PerImageCheckpointMetrics(
            metric_name="relative_error",
            metric_value=metric_value,
        )
    raise ValueError(
        f"Unsupported per-image attribute network: {args.network_type}"
    )
