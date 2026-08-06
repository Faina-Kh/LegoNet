"""Pure metrics for single-label classification attributes."""

from typing import Any, Optional, Sequence

import numpy as np

from legonet.eval.per_object_result import (
    ClassificationMetrics,
    ClassificationType,
    ClassMetrics,
)


def decode_class_predictions(
    predictions: Any,
    classification_type: ClassificationType,
    threshold: float = 0.5,
) -> np.ndarray:
    """Decode per-sample scores into class indices.

    Binary predictions may be scalar scores or a two-column score matrix.
    Nominal and ordinal predictions use the highest-scoring class per sample.
    """
    values = np.asarray(predictions)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim == 1:
        if classification_type is not ClassificationType.BINARY:
            raise ValueError("Multiclass predictions require one score per class.")
        return (values >= threshold).astype(int)
    if values.ndim != 2:
        raise ValueError("Predictions must be a score vector or matrix.")
    if values.shape[1] == 1:
        if classification_type is not ClassificationType.BINARY:
            raise ValueError("Multiclass predictions require multiple columns.")
        return (values[:, 0] >= threshold).astype(int)
    return np.argmax(values, axis=1)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    """Return a ratio, or ``None`` when it is undefined."""
    return numerator / denominator if denominator > 0 else None


def _mean_available(values: Sequence[Optional[float]]) -> Optional[float]:
    """Average defined metric values."""
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def compute_classification_metrics(
    ground_truth: Sequence[Any],
    predictions: Sequence[Any],
    class_labels: Sequence[Any],
    class_names: Optional[Sequence[str]] = None,
    classification_type: ClassificationType = ClassificationType.NOMINAL,
    eligible_samples: Optional[int] = None,
) -> ClassificationMetrics:
    """Calculate confusion-based metrics for aligned single-label outputs."""
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have equal length.")
    if len(class_labels) == 0 or len(set(class_labels)) != len(class_labels):
        raise ValueError("Class labels must be nonempty and unique.")
    if class_names is None:
        names = tuple(str(label) for label in class_labels)
    else:
        names = tuple(class_names)
        if len(names) != len(class_labels):
            raise ValueError("Class labels and class names must have equal length.")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique.")

    evaluated_samples = len(ground_truth)
    if eligible_samples is None:
        eligible_samples = evaluated_samples
    if eligible_samples < evaluated_samples:
        raise ValueError("Eligible samples cannot be fewer than evaluated samples.")

    label_indices = {label: index for index, label in enumerate(class_labels)}
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    true_indices = []
    predicted_indices = []
    for target, prediction in zip(ground_truth, predictions):
        if target not in label_indices or prediction not in label_indices:
            raise ValueError("Observed labels must be present in class_labels.")
        target_index = label_indices[target]
        prediction_index = label_indices[prediction]
        confusion[target_index, prediction_index] += 1
        true_indices.append(target_index)
        predicted_indices.append(prediction_index)

    correct = int(np.trace(confusion))
    accuracy = _safe_ratio(correct, evaluated_samples)
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    for index, name in enumerate(names):
        true_positive = int(confusion[index, index])
        false_positive = int(np.sum(confusion[:, index]) - true_positive)
        false_negative = int(np.sum(confusion[index, :]) - true_positive)
        support = int(np.sum(confusion[index, :]))
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1_score = _safe_ratio(2 * true_positive, f1_denominator)
        per_class[name] = ClassMetrics(support, precision, recall, f1_score)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    one_minus_fvu = None
    if (
        evaluated_samples > 0
        and classification_type
        in (ClassificationType.BINARY, ClassificationType.ORDINAL)
    ):
        targets = np.asarray(true_indices, dtype=float)
        estimates = np.asarray(predicted_indices, dtype=float)
        variance = float(np.var(targets))
        if variance > 0:
            mean_squared_error = float(np.mean((targets - estimates) ** 2))
            one_minus_fvu = 1.0 - mean_squared_error / variance

    confusion_tuple = tuple(
        tuple(int(value) for value in row) for row in confusion
    )
    return ClassificationMetrics(
        classification_type=classification_type,
        class_names=names,
        confusion_matrix=confusion_tuple,
        eligible_samples=eligible_samples,
        evaluated_samples=evaluated_samples,
        correct_predictions=correct,
        accuracy=accuracy,
        balanced_accuracy=_mean_available(recalls),
        macro_precision=_mean_available(precisions),
        macro_recall=_mean_available(recalls),
        macro_f1=_mean_available(f1_scores),
        micro_precision=accuracy,
        micro_recall=accuracy,
        micro_f1=accuracy,
        per_class=per_class,
        one_minus_fvu=one_minus_fvu,
    )
