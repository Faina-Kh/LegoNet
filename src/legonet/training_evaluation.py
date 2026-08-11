"""Evaluation helpers used during LegoNet training."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np

from legonet import config
from legonet.eval import detection_eval
from legonet.eval import perObject_eval


@dataclass(frozen=True)
class DetectionMetrics:
    """Detection metrics produced by an in-training evaluation."""

    mean_average_precision: float
    precision: float
    recall: float


@dataclass(frozen=True)
class IoURelativeError:
    """Attribute error and detection diagnostics at one IoU threshold."""

    iou_threshold: float
    relative_error: Optional[float]
    matched_objects: Optional[int] = None
    recall: Optional[float] = None
    precision: Optional[float] = None


@dataclass(frozen=True)
class IoUSweepResult:
    """Relative-error measurements collected across IoU thresholds."""

    measurements: Tuple[IoURelativeError, ...]

    @property
    def average_relative_error(self) -> Optional[float]:
        """Return the mean of valid errors, or ``None`` when none exist."""
        valid_errors = [
            measurement.relative_error
            for measurement in self.measurements
            if measurement.relative_error is not None
        ]
        if not valid_errors:
            return None
        return float(np.mean(valid_errors))


@dataclass(frozen=True)
class CheckpointSelectionMetrics:
    """Task-neutral metrics used during checkpoint selection."""

    metric_name: str
    metric_value: Optional[float]
    one_minus_fvu: Optional[float]


ATTRIBUTE_CHECKPOINT_NAMES = {"length", "diameter", "color"}


def _finite_metric(value: Any) -> Optional[float]:
    """Convert one finite metric value or return ``None`` when unavailable."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def select_checkpoint_metrics(
    metrics: Any,
    *,
    network_type: str,
    requested_attribute: str | None,
) -> CheckpointSelectionMetrics:
    """Select the fixed error metric for counting or one chosen attribute."""
    if network_type == "per_object_counting":
        metric_name = "count_relative_error"
        metric_value = metrics.count_relative_error
        one_minus_fvu = _finite_metric(1 - metrics.count_fvu)
    else:
        attribute_name = requested_attribute or "length"
        if attribute_name not in ATTRIBUTE_CHECKPOINT_NAMES:
            raise ValueError(
                f"Checkpoint attribute {attribute_name!r} is not valid for "
                f"{network_type}. Choose one of: "
                f"{', '.join(sorted(ATTRIBUTE_CHECKPOINT_NAMES))}."
            )
        color_metrics = metrics.color_metrics
        if attribute_name == "length":
            metric_name = "length_relative_error"
            metric_value = metrics.trl_relative_error
            one_minus_fvu = _finite_metric(1 - metrics.trl_fvu)
        elif attribute_name == "diameter":
            metric_name = "diameter_relative_error"
            metric_value = metrics.diameter_relative_error
            one_minus_fvu = _finite_metric(1 - metrics.diameter_fvu)
        else:
            metric_name = "color_error_rate"
            metric_value = (
                color_metrics.error_rate if color_metrics is not None else None
            )
            one_minus_fvu = _finite_metric(
                color_metrics.one_minus_fvu
                if color_metrics is not None
                else None
            )

    if requested_attribute and network_type == "per_object_counting":
        raise ValueError(
            "Per-object counting always uses count relative error and does not "
            "accept a checkpoint attribute."
        )
    has_ground_truth = getattr(metrics, "has_original_gt", True)
    return CheckpointSelectionMetrics(
        metric_name=metric_name,
        metric_value=(
            _finite_metric(metric_value) if has_ground_truth else None
        ),
        one_minus_fvu=one_minus_fvu if has_ground_truth else None,
    )


@contextmanager
def _validation_dataset(model: Any, dataset_val: Any) -> Iterator[None]:
    """Temporarily point a model at validation data and always restore it."""
    training_dataset = model.dataset
    model.dataset = dataset_val
    try:
        yield
    finally:
        model.dataset = training_dataset


def _relative_error(output: Sequence[Any]) -> Optional[float]:
    """Extract a valid relative error from evaluator output."""
    if len(output) == 0 or output[0] == -1:
        return None
    return output[0]


def _iou_measurement(
    iou_threshold: float,
    output: Sequence[Any],
) -> IoURelativeError:
    """Map evaluator output to one threshold's error and detection diagnostics."""
    has_detection_diagnostics = len(output) >= 5
    return IoURelativeError(
        iou_threshold=iou_threshold,
        relative_error=_relative_error(output),
        matched_objects=int(output[2]) if has_detection_diagnostics else None,
        recall=float(output[3]) if has_detection_diagnostics else None,
        precision=float(output[4]) if has_detection_diagnostics else None,
    )


def evaluate_detection(
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
) -> DetectionMetrics:
    """Evaluate detection with the currently configured score and IoU limits."""
    model.eval()
    mean_average_precision, precision, recall = detection_eval.evaluateMAP_simple(
        dataset_val,
        dataloader_val,
        sampler_val,
        model,
        score_threshold=config.Detection.min_score,
        iou_threshold=config.Detection.iou_threshold,
    )
    return DetectionMetrics(mean_average_precision, precision, recall)


def evaluate_per_object_checkpoint_metrics(
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
    args: Any,
) -> CheckpointSelectionMetrics:
    """Evaluate one IoU and return metrics used to select a checkpoint."""
    model.eval()
    with _validation_dataset(model, dataset_val):
        metrics = perObject_eval.eval(
            dataset_val,
            dataloader_val,
            sampler_val,
            model,
            verbose=False,
            to_draw=False,
            print_to_files=True,
            args=args,
            return_metrics=True,
        )
    return select_checkpoint_metrics(
        metrics,
        network_type=args.network_type,
        requested_attribute=getattr(args, "checkpoint_attribute", None),
    )


def evaluate_combined_iou_sweep(
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
    args: Any,
    iou_thresholds: Sequence[float],
) -> IoUSweepResult:
    """Evaluate combined tasks across IoUs while restoring mutable run state."""
    model.eval()
    original_iou_threshold = config.Detection.iou_threshold
    measurements = []
    try:
        with _validation_dataset(model, dataset_val):
            for iou_threshold in iou_thresholds:
                config.Detection.iou_threshold = iou_threshold
                output = perObject_eval.eval(
                    dataset_val,
                    dataloader_val,
                    sampler_val,
                    model,
                    verbose=False,
                    to_draw=False,
                    print_to_files=True,
                    args=args,
                )
                measurements.append(_iou_measurement(iou_threshold, output))
            print()
    finally:
        config.Detection.iou_threshold = original_iou_threshold

    return IoUSweepResult(tuple(measurements))
