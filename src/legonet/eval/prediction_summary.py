"""Per-image prediction summary bookkeeping for evaluation."""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from legonet.eval.classification_metrics import decode_class_predictions
from legonet.eval.per_object_result import ClassificationType


def record_per_image_predictions(
    state: MutableMapping[str, Any],
    image_name: str,
    estimation_outputs,
    *,
    attributes: bool,
) -> None:
    """Record object predictions and aggregate them for one image."""
    if estimation_outputs is None:
        return

    predictions = estimation_outputs[0]
    predicted_count_sum = 0
    predicted_trl_sum = 0
    predicted_diameter_sum = 0
    for prediction in predictions:
        if attributes:
            predicted_count = np.round(prediction.cpu()[0].numpy())
            predicted_length = prediction.cpu()[1].numpy()
            predicted_diameter = prediction.cpu()[2].numpy()
            predicted_color = decode_class_predictions(
                [prediction.cpu()[0].item()], ClassificationType.BINARY
            )[0]
            state.setdefault("predicted_lengths_any_crop", []).append(predicted_length)
            state.setdefault("predicted_dia_any_crop", []).append(predicted_diameter)
            state.setdefault("predicted_color_any_crop", []).append(predicted_color)
            predicted_trl_sum += predicted_length
            predicted_diameter_sum += predicted_diameter
        else:
            predicted_count = np.round(prediction.cpu().item())

        state["predicted_counts_any_crop"].append(predicted_count)
        predicted_count_sum += predicted_count

    number_of_predictions = predictions.shape[0]
    predicted_average = predicted_count_sum / number_of_predictions
    state["per_im_pred_avg"].append(predicted_average)
    state["per_im_pred_dict"][image_name] = predicted_average

    if attributes:
        state.setdefault("TRL_per_im_pred_sum", []).append(predicted_trl_sum)
        state.setdefault("TRL_per_im_pred_dict", {})[image_name] = predicted_trl_sum
        predicted_diameter_average = predicted_diameter_sum / number_of_predictions
        state.setdefault("dia_per_im_pred_avg", []).append(predicted_diameter_average)
        state.setdefault("dia_per_im_pred_dict", {})[image_name] = predicted_diameter_average
