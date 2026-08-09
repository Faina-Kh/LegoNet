"""Evaluation-state construction for per-object inference."""

from __future__ import annotations

from typing import Any, Optional

from legonet import config


def initiate_global_dicts(
    state: Optional[dict[str, Any]] = None,
    image_name: str = "",
    initiate: bool = False,
) -> dict[str, Any]:
    """Create evaluation state or add records for one image.

    Args:
        state: Existing state when adding per-image records.
        image_name: Image name to initialize when ``initiate`` is false.
        initiate: Create and return a fresh state when true.

    Returns:
        The newly created or extended evaluation state.
    """
    is_counting = config.Detect_and_Estimate.type == "per_object_counting"
    is_attributes = config.Detect_and_Estimate.type in {
        "per_object_attributes",
        "per_object_attributes_multibranch",
    }

    if initiate:
        state = {
            "all_predicted_counts": [],
            "T": [],
            "P": [],
            "all_crops_GT_counts": [],
            "all_orig_GT_counts": [],
            "orig_abs_diff": [],
            "orig_rel_error": [],
            "all_data_gt_count": [],
            "gt_objects_withGTpoints": 0,
            "found_orig_objects": 0,
            "FP": 0,
            "predicted_counts_any_crop": [],
            "matched_without_gt_points": 0,
            "crops_without_gt_points": 0,
            "detections_data_any_crop": {},
            "not_found_gt": {},
            "no_predictions": {},
            "per_im_gt_avg": [],
            "per_im_gt_avg_dict": {},
            "per_im_pred_avg": [],
            "per_im_pred_dict": {},
            "num_of_gt_boxes": 0,
            "all_detections": [],
            "all_annotations": [],
        }

        if is_attributes:
            state.update(
                {
                    "all_predicted_TRL": [],
                    "all_predicted_dia": [],
                    "all_predicted_color": [],
                    "all_crops_GT_TRL": [],
                    "crops_abs_diff_TRL": [],
                    "crops_rel_error_TRL": [],
                    "all_crops_GT_dia": [],
                    "crops_abs_diff_dia": [],
                    "crops_rel_error_dia": [],
                    "all_orig_GT_TRL": [],
                    "orig_abs_diff_TRL": [],
                    "orig_rel_error_TRL": [],
                    "all_orig_GT_dia": [],
                    "orig_abs_diff_dia": [],
                    "orig_rel_error_dia": [],
                    "all_orig_GT_color": [],
                    "orig_abs_diff_color": [],
                    "all_data_gt_TRL": [],
                    "predicted_TRL_any_crop": [],
                    "all_data_gt_dia": [],
                    "predicted_dia_any_crop": [],
                    "all_data_gt_color": [],
                    "predicted_color_any_crop": [],
                    "TRL_per_im_gt_sum": [],
                    "TRL_per_im_gt_sum_dict": {},
                    "TRL_per_im_pred_sum": [],
                    "TRL_per_im_pred_dict": {},
                    "dia_per_im_gt_avg": [],
                    "dia_per_im_gt_avg_dict": {},
                    "dia_per_im_pred_avg": [],
                    "dia_per_im_pred_dict": {},
                }
            )
        return state

    if state is None:
        raise ValueError("state must be provided when initiate=False")
    if not image_name:
        raise ValueError("image_name must be provided when initiate=False")

    state["detections_data_any_crop"][image_name] = {
        "gt_count": [],
        "label": [],
        "score": [],
        "gt_box_id": [],
        "max_overlap": [],
    }
    state["not_found_gt"][image_name] = {
        "gt_count": [],
        "label": [],
        "score": [],
        "max_overlap": [],
    }

    if is_counting:
        state["detections_data_any_crop"][image_name]["pred"] = []
        state["not_found_gt"][image_name]["pred"] = []

    if is_attributes:
        attribute_fields = {
            "color_pred": [],
            "color_gt": [],
            "TRL_pred": [],
            "TRL_gt": [],
            "dia_pred": [],
            "dia_gt": [],
        }
        state["detections_data_any_crop"][image_name].update(
            {key: value.copy() for key, value in attribute_fields.items()}
        )
        state["not_found_gt"][image_name].update(
            {key: value.copy() for key, value in attribute_fields.items()}
        )

    return state
