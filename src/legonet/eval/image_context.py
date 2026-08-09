"""Per-image input preparation for per-object evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class EvaluationImageContext:
    """Dataset metadata and annotations needed to evaluate one image."""

    image: torch.Tensor
    scale: Any
    group_indices: Any
    image_id: Any
    image_name: str
    image_path: str
    box_annotations: Any | None
    count_annotations: Any


def prepare_image_context(
    dataset: Any,
    sampler: Any,
    iteration: int,
    data: Mapping[str, Any],
    *,
    have_gt: bool,
    all_box_annotations: Sequence[Any] | None = None,
    all_count_annotations: Sequence[Any] | None = None,
) -> EvaluationImageContext:
    """Resolve one dataloader item to its dataset identity and GT records."""
    group_indices = sampler.groups[iteration]
    image_id = dataset.image_ids[group_indices[0]]
    image_name = dataset.img_info[image_id]["name"]

    if have_gt:
        if all_box_annotations is None or all_count_annotations is None:
            raise ValueError("Ground-truth annotation collections are required")
        box_annotations = all_box_annotations[image_id][0]
        count_annotations = all_count_annotations[image_id][0]
    else:
        box_annotations = None
        count_annotations = []

    return EvaluationImageContext(
        image=data["img"].clone().detach(),
        scale=data["scale"],
        group_indices=group_indices,
        image_id=image_id,
        image_name=image_name,
        image_path=os.path.join(dataset.base_dir, image_name),
        box_annotations=box_annotations,
        count_annotations=count_annotations,
    )
