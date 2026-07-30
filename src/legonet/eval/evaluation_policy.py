"""Image-inclusion policies shared by LegoNet evaluators."""

from enum import Enum
from typing import Sized


class EvaluationTask(Enum):
    """Evaluation tasks with distinct ground-truth requirements."""

    DETECTION = "detection"
    PER_OBJECT = "per_object"


def should_include_image(
    task: EvaluationTask,
    box_annotations: Sized,
    per_object_annotations: Sized,
) -> bool:
    """Return whether an image has the targets required by ``task``."""
    if task is EvaluationTask.DETECTION:
        return True
    if task is EvaluationTask.PER_OBJECT:
        return len(box_annotations) > 0 and len(per_object_annotations) > 0
    raise ValueError(f"Unsupported evaluation task: {task!r}")
