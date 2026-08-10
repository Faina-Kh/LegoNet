"""Bookkeeping for evaluation images without predicted boxes."""

from __future__ import annotations

from typing import Any, MutableMapping


def record_no_predictions(
    state: MutableMapping[str, Any],
    image_name: str,
    *,
    image_gt_average: float,
    image_trl_sum: float,
    image_diameter_average: float,
    attributes: bool,
    network_type: str,
) -> None:
    """Append the legacy sentinel record for an image with no detections."""
    if image_name not in state["no_predictions"]:
        record = {
            "gt_count": [],
            "label": [],
            "score": [],
            "max_overlap": [],
        }
        if network_type == "per_object_counting":
            record["pred"] = []
        if attributes:
            record.update(
                {
                    "TRL_pred": [],
                    "TRL_gt": [],
                    "dia_pred": [],
                    "dia_gt": [],
                    "color_pred": [],
                    "color_gt": [],
                }
            )
        state["no_predictions"][image_name] = record

    record = state["no_predictions"][image_name]
    if network_type == "per_object_counting":
        record["pred"].append(0)
    record["gt_count"].append(image_gt_average)
    record["label"].append(-1)
    record["score"].append(-1)
    record["max_overlap"].append(-1)

    if attributes:
        record["TRL_pred"].append(0)
        record["TRL_gt"].append(image_trl_sum)
        record["dia_pred"].append(0)
        record["dia_gt"].append(image_diameter_average)
        record["color_pred"].append(-1)
        record["color_gt"].append(
            -1 if image_trl_sum == 0 else image_gt_average
        )
