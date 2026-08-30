"""Typed contract for legacy per-object evaluation results.

The legacy evaluator currently returns one of two positional sequences:

* counting: nine values containing count and detection metrics;
* attributes: one value containing total-root-length relative error.

This module documents that boundary without changing the evaluator itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class PerObjectResultKind(Enum):
    """Supported legacy result layouts."""

    COUNTING = "counting"
    ATTRIBUTES = "attributes"


class ClassificationType(Enum):
    """Semantic classification types supported by attribute evaluation."""

    BINARY = "binary"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class RegressionMetrics:
    """Metrics for one continuous-valued attribute."""

    eligible_samples: Optional[int] = None
    evaluated_samples: Optional[int] = None
    mean_absolute_error: Optional[float] = None
    mean_squared_error: Optional[float] = None
    mean_relative_error: Optional[float] = None
    one_minus_fvu: Optional[float] = None

    @property
    def coverage(self) -> Optional[float]:
        """Return the evaluated fraction, if an eligible population exists."""
        if not self.eligible_samples:
            return None
        if self.evaluated_samples is None:
            return None
        return self.evaluated_samples / self.eligible_samples


@dataclass(frozen=True)
class ClassMetrics:
    """One-vs-rest metrics for a single classification label."""

    support: int
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics for binary, nominal, or ordinal classification attributes.

    ``class_names`` defines the row and column order of ``confusion_matrix``.
    Classification 1-FVU is meaningful only for binary or ordinal labels and
    remains ``None`` for nominal labels.
    """

    classification_type: ClassificationType
    class_names: Tuple[str, ...]
    confusion_matrix: Tuple[Tuple[int, ...], ...]
    eligible_samples: Optional[int] = None
    evaluated_samples: Optional[int] = None
    correct_predictions: Optional[int] = None
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    micro_precision: Optional[float] = None
    micro_recall: Optional[float] = None
    micro_f1: Optional[float] = None
    per_class: Mapping[str, ClassMetrics] = field(default_factory=dict)
    one_minus_fvu: Optional[float] = None

    def __post_init__(self) -> None:
        """Reject confusion matrices whose dimensions lack clear semantics."""
        class_count = len(self.class_names)
        if len(self.confusion_matrix) != class_count or any(
            len(row) != class_count for row in self.confusion_matrix
        ):
            raise ValueError(
                "Confusion matrix dimensions must match the class-name order."
            )

    @property
    def coverage(self) -> Optional[float]:
        """Return the classified fraction, if eligible samples are known."""
        if not self.eligible_samples:
            return None
        if self.evaluated_samples is None:
            return None
        return self.evaluated_samples / self.eligible_samples

    @property
    def error_rate(self) -> Optional[float]:
        """Return the fraction of evaluated predictions classified incorrectly."""
        return None if self.accuracy is None else 1.0 - self.accuracy


@dataclass(frozen=True)
class AttributeEvaluationResult:
    """Metrics grouped by configured attribute name and prediction task."""

    regression: Mapping[str, RegressionMetrics] = field(default_factory=dict)
    classification: Mapping[str, ClassificationMetrics] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class PerObjectEvaluationResult:
    """Named representation of the positional ``perObject_eval.eval`` output.

    Counting results populate every field. Attribute results populate only
    ``relative_error``; in that mode it describes per-object root length.
    Values are intentionally not normalized so conversion preserves legacy
    Python and NumPy scalar types exactly.
    """

    kind: PerObjectResultKind
    relative_error: Any
    ground_truth_objects: Optional[Any] = None
    matched_objects: Optional[Any] = None
    recall: Optional[Any] = None
    precision: Optional[Any] = None
    mean_absolute_error: Optional[Any] = None
    exact_count_agreement: Optional[Any] = None
    mean_squared_error: Optional[Any] = None
    one_minus_fvu: Optional[Any] = None
    attributes: Optional[AttributeEvaluationResult] = None

    @classmethod
    def from_legacy_sequence(
        cls, output: Sequence[Any]
    ) -> "PerObjectEvaluationResult":
        """Create a named result from a documented legacy layout."""
        if len(output) == 1:
            length_metrics = RegressionMetrics(mean_relative_error=output[0])
            return cls(
                kind=PerObjectResultKind.ATTRIBUTES,
                relative_error=output[0],
                attributes=AttributeEvaluationResult(
                    regression={"length": length_metrics}
                ),
            )
        if len(output) == 9:
            return cls(
                kind=PerObjectResultKind.COUNTING,
                relative_error=output[0],
                ground_truth_objects=output[1],
                matched_objects=output[2],
                recall=output[3],
                precision=output[4],
                mean_absolute_error=output[5],
                exact_count_agreement=output[6],
                mean_squared_error=output[7],
                one_minus_fvu=output[8],
            )
        raise ValueError(
            "Per-object evaluation output must contain 1 attribute metric "
            f"or 9 counting metrics; received {len(output)} values."
        )

    def to_legacy_tuple(self) -> Tuple[Any, ...]:
        """Return the historical positional result layout."""
        if self.kind is PerObjectResultKind.ATTRIBUTES:
            return (self.relative_error,)
        return (
            self.relative_error,
            self.ground_truth_objects,
            self.matched_objects,
            self.recall,
            self.precision,
            self.mean_absolute_error,
            self.exact_count_agreement,
            self.mean_squared_error,
            self.one_minus_fvu,
        )
