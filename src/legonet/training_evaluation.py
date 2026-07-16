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


def evaluate_combined_once(
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
    args: Any,
) -> Optional[float]:
    """Evaluate combined tasks once while preserving the training dataset."""
    model.eval()
    with _validation_dataset(model, dataset_val):
        output = perObject_eval.eval(
            dataset_val,
            dataloader_val,
            sampler_val,
            model,
            to_draw=False,
            print_to_files=True,
            args=args,
        )
    return _relative_error(output)


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
