"""Model-input construction for per-object evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def build_per_object_model_input(
    image: torch.Tensor,
    data: Mapping[str, Any],
    group_indices: Any,
    *,
    have_gt: bool,
    device: Any,
) -> list[Any]:
    """Build model input for inference without GT or estimation evaluation with GT."""
    model_image = image.to(device).float()
    group_tensor = torch.tensor(group_indices)
    if not have_gt:
        annotations = None
    else:
        annotations = [data["bbox_annot"], data.get("points_annot")]
    return [model_image, annotations, group_tensor]
