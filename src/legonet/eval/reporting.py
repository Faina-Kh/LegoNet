"""CSV artifact writers for LegoNet evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from legonet.eval.per_object_result import ClassificationMetrics


COUNT_COLUMNS = (
    "img", "crop", "gt_count", "pred_count", "label", "score",
    "max_overlap",
)
ATTRIBUTE_COLUMNS = (
    "img", "crop", "gt_color", "pred_color", "label", "score",
    "max_overlap", "gt_TRL", "pred_TRL", "gt_dia", "pred_dia",
)
COUNT_SUMMARY_COLUMNS = (
    "img", "gt_count", "pred", "label", "score", "max_overlap",
)
ATTRIBUTE_SUMMARY_COLUMNS = (
    "img", "gt_color", "pred_color", "label", "score", "max_overlap",
    "gt_TRL", "pred_TRL", "gt_dia", "pred_dia",
)
SEPARATOR = "=" * 100


def _metric_text(value: Any, precision: int = 3) -> str:
    """Format a finite metric or report it as unavailable."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def format_counting_summary(
    mean_absolute_error: float,
    exact_agreement: float,
    mean_squared_error: float,
    mean_relative_error: float,
    one_minus_fvu: float,
    crop_metrics: Mapping[str, Any],
) -> str:
    """Format global counting and positive-crop diagnostics."""
    return (
        f"{SEPARATOR}\n"
        "Evaluation Summary - per-object counting for IoU-matched GT boxes\n"
        f"orig_avg_abs_count_diff: {mean_absolute_error:.3f} | "
        f"orig_count_agreement: {exact_agreement:.3f} | "
        f"orig_MSE: {mean_squared_error:.3f} | "
        f"orig_avg_relative_error: {mean_relative_error:.3f} | "
        f"orig_1-FVU: {one_minus_fvu:f}\n\n"
        f"{SEPARATOR}\n"
        "Diagnostic - counting within positive predicted crops\n"
        f"positive_crops: {crop_metrics['num_positive']} | "
        f"empty_crops: {crop_metrics['num_empty']} | "
        f"total_evaluated_crops: {crop_metrics['num_total']}\n"
        f"crop_MAE: {crop_metrics['mae']:.3f} | "
        f"crop_count_agreement: {crop_metrics['exact_agreement']:.3f} | "
        f"crop_MSE: {crop_metrics['mse']:.3f} | "
        f"crop_avg_relative_error: {crop_metrics['mean_relative_error']:.3f}\n"
    )


def format_attribute_summary(
    trl_mean_absolute_error: float,
    trl_mean_squared_error: float,
    trl_mean_relative_error: float,
    trl_one_minus_fvu: float,
    diameter_mean_absolute_error: float,
    diameter_mean_squared_error: float,
    diameter_mean_relative_error: float,
    diameter_one_minus_fvu: float,
    color_metrics: ClassificationMetrics | None,
) -> str:
    """Format global roots regression and color classification metrics."""
    report = (
        f"{SEPARATOR}\n"
        "Evaluation Summary - per-object attributes\n"
        f"orig_avg_abs_TRL_diff: {trl_mean_absolute_error:.3f} | "
        f"orig_MSE_TRL: {trl_mean_squared_error:.3f} | "
        f"orig_avg_relative_error_TRL: {trl_mean_relative_error:.3f} | "
        f"orig_1-FVU_TRL: {trl_one_minus_fvu:f}\n"
        f"orig_avg_abs_dia_diff: {diameter_mean_absolute_error:.3f} | "
        f"orig_MSE_dia: {diameter_mean_squared_error:.3f} | "
        f"orig_avg_relative_error_dia: {diameter_mean_relative_error:.3f} | "
        f"orig_1-FVU_dia: {diameter_one_minus_fvu:f}\n"
    )
    if color_metrics is not None:
        report += (
            f"color_classes: {color_metrics.class_names} | "
            #f"color_correct: {color_metrics.correct_predictions} | "
            #f"color_evaluated: {color_metrics.evaluated_samples} | "
            #f"color_accuracy: {_metric_text(color_metrics.accuracy)} | "
            f"color_error_rate: {_metric_text(color_metrics.error_rate)} | "
            f"color_1-FVU: {_metric_text(color_metrics.one_minus_fvu)} | "
            #"color_balanced_accuracy: "
            #f"{_metric_text(color_metrics.balanced_accuracy)} | "
            #f"color_macro_F1: {_metric_text(color_metrics.macro_f1)} | "
            #f"color_coverage: {_metric_text(color_metrics.coverage)}\n"
            #"color_macro_precision: "
            #f"{_metric_text(color_metrics.macro_precision)} | "
            #f"color_macro_recall: {_metric_text(color_metrics.macro_recall)} | "
            f"color_confusion_matrix: {color_metrics.confusion_matrix}\n"
        )
    return report + "\n"


def format_keypoint_summary(
    average_precision: float,
    recall: float,
    precision: float,
) -> str:
    """Format the aggregate keypoint-detection metrics."""
    return (
        f"{SEPARATOR}\n"
        "Evaluation Summary - keypoint detection\n"
        f"mAP: {_metric_text(average_precision)} | "
        f"recall: {_metric_text(recall)} | "
        f"precision: {_metric_text(precision)}\n"
    )


def format_matching_diagnostics(
    scope: str,
    ground_truth_objects: int,
    matched_objects: int,
    false_positives: int,
) -> str:
    """Format scope-explicit object matching counts."""
    recall = matched_objects / ground_truth_objects if ground_truth_objects else 0.0
    prediction_count = matched_objects + false_positives
    precision = matched_objects / prediction_count if prediction_count else 0.0
    return (
        f"{SEPARATOR}\n"
        f"Object matching diagnostics ({scope})\n"
        f"GT objects considered: {ground_truth_objects}\n"
        f"IoU-matched objects: {matched_objects} ({100 * recall:.2f}% recall)\n"
        f"Unmatched or duplicate predictions: {false_positives}\n"
        f"Matched-object precision: {100 * precision:.2f}% "
        f"({matched_objects} / ({matched_objects} + {false_positives}))\n\n"
    )


def format_detection_metrics(metrics: Sequence[Any]) -> str:
    """Format full-dataset bounding-box detection metrics."""
    mean_ap, precision, recall = metrics[:3]
    return (
        f"{SEPARATOR}\n"
        "Bounding-box detection stats (all images)\n"
        f"mAP = {mean_ap:.3f} | precision = {precision:.3f} | "
        f"recall = {recall:.3f}\n\n"
    )


def format_counting_per_image(
    ground_truth: Mapping[str, float], predictions: Mapping[str, float]
) -> str:
    """Format legacy per-image counting values and relative errors."""
    lines = [
        "Per image stats based on IoU-matched GT boxes",
        "Per image gt, predicted avg count",
    ]
    errors = []
    for image_name, prediction in predictions.items():
        target = ground_truth[image_name]
        error = abs(target - prediction) / target
        errors.append(error)
        lines.append(
            f"{image_name}: avg_gt: {target:.2f}, avg_pred: {prediction:.2f}, "
            f"rel_error: {error:.2f}"
        )
    lines.append(f"Avg of per image rel_error:{np.mean(errors):.4f}")
    return "\n".join(lines) + "\n"


def format_prediction_aggregates(
    predictions: Mapping[str, float],
    *,
    trl_predictions: Mapping[str, float] | None = None,
    diameter_predictions: Mapping[str, float] | None = None,
    images_without_detections: Sequence[str] = (),
) -> str:
    """Format per-image aggregates when ground truth is unavailable."""
    lines = ["Per-image predicted aggregates"]
    if trl_predictions is None or diameter_predictions is None:
        for image_name, prediction in predictions.items():
            lines.append(
                f"{image_name}: predicted average count: {prediction:.2f}"
            )
    else:
        for image_name, color_prediction in predictions.items():
            lines.append(
                f"{image_name}: predicted length sum (TRL): "
                f"{trl_predictions[image_name]:.2f}, predicted diameter average: "
                f"{diameter_predictions[image_name]:.2f}, "
                f"predicted color average: {color_prediction:.2f}"
            )
    for image_name in images_without_detections:
        if image_name not in predictions:
            lines.append(f"{image_name}: no detected objects")
    return "\n".join(lines) + "\n"


def format_roots_per_image(
    trl_ground_truth: Mapping[str, float],
    trl_predictions: Mapping[str, float],
    diameter_ground_truth: Mapping[str, float],
    diameter_predictions: Mapping[str, float],
    color_ground_truth: Mapping[str, float],
    color_predictions: Mapping[str, float],
    per_image_metrics: Any,
) -> str:
    """Format image-level roots aggregates and their dataset metrics."""
    image_names = list(color_predictions)
    lines = [
        "Per image stats based on IoU-matched GT boxes",
        "Per image gt TRL (sum of RL), predicted sum TRL",
    ]
    trl_errors = []
    for image_name, prediction in trl_predictions.items():
        target = trl_ground_truth[image_name]
        error = abs(target - prediction) / target if target > 0 else -1
        trl_errors.append(error)
        lines.append(
            f"{image_name}: sum_gt_TRL: {target:.2f}, pred_TRL: "
            f"{prediction:.2f}, rel_error_TRL: {error:.2f}"
        )
    valid_trl_errors = [error for error in trl_errors if error > 0]
    lines.extend(
        [
            "Avg of per image rel_error of TRL:"
            f"{np.mean(valid_trl_errors):.4f} | 1-FVU: "
            f"{_metric_text(per_image_metrics.trl.one_minus_fvu, 4)}",
            "",
        ]
    )

    diameter_errors = []
    diameter_differences = []
    for image_name, prediction in diameter_predictions.items():
        target = diameter_ground_truth[image_name]
        difference = abs(target - prediction)
        error = difference / target if target > 0 else -1
        diameter_differences.append(difference)
        diameter_errors.append(error)
        lines.append(
            f"{image_name}: avg_gt_dia: {target:.2f}, avg_pred_dia: "
            f"{prediction:.2f}, rel_error_dia: {error:.2f}"
        )
    lines.extend(
        [
            "Avg of per image rel_error of diameter:"
            f"{np.mean(diameter_errors):.4f} | 1-FVU: "
            f"{_metric_text(per_image_metrics.diameter.one_minus_fvu, 4)} |"
            f"abs difference:{np.mean(diameter_differences):.4f}",
            "",
            "Per image average GT and predicted color",
        ]
    )

    color_differences = []
    for image_name in image_names:
        target = color_ground_truth[image_name]
        prediction = color_predictions[image_name]
        difference = abs(target - prediction)
        color_differences.append(difference)
        lines.append(
            f"{image_name}: avg_gt_color: {target:.2f}, avg_pred_color: "
            f"{prediction:.2f}, abs_error_color: {difference:.2f}"
        )
    lines.append(
        "Avg of per image absolute error of color: "
        f"{_metric_text(float(np.mean(color_differences)) if color_differences else None, 4)} "
        f"| 1-FVU: {_metric_text(per_image_metrics.color.one_minus_fvu, 4)}"
    )
    return "\n".join(lines) + "\n"


def _write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Write one CSV with a header and prebuilt rows."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        writer.writerows(rows)


def _detection_rows(
    records: Mapping[str, Mapping[str, Sequence[Any]]],
    attributes: bool,
) -> list[list[Any]]:
    """Build rows for every predicted crop."""
    rows = []
    for image_name, image_records in records.items():
        for index in range(len(image_records["score"])):
            if attributes:
                row = [
                    image_name,
                    index,
                    image_records["color_gt"][index],
                    image_records["color_pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                    image_records["TRL_gt"][index],
                    image_records["TRL_pred"][index],
                    image_records["dia_gt"][index],
                    image_records["dia_pred"][index],
                ]
            else:
                row = [
                    image_name,
                    index,
                    image_records["gt_count"][index],
                    image_records["pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                ]
            rows.append(row)
    return rows


def _summary_rows(
    records: Mapping[str, Mapping[str, Sequence[Any]]],
    attributes: bool,
    empty_image_placeholders: bool,
) -> list[list[Any]]:
    """Build unmatched-GT or no-prediction rows using legacy placeholders."""
    rows = []
    prediction_key = "TRL_pred" if attributes else "pred"
    for image_name, image_records in records.items():
        predictions = image_records[prediction_key] if image_records else ()
        for index in range(len(predictions)):
            if attributes:
                row = [
                    image_name,
                    image_records["color_gt"][index],
                    image_records["color_pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                    image_records["TRL_gt"][index],
                    image_records["TRL_pred"][index],
                    image_records["dia_gt"][index],
                    image_records["dia_pred"][index],
                ]
            else:
                row = [
                    image_name,
                    image_records["gt_count"][index],
                    image_records["pred"][index],
                    image_records["label"][index],
                    image_records["score"][index],
                    image_records["max_overlap"][index],
                ]
            rows.append(row)

        if predictions:
            continue
        if empty_image_placeholders:
            rows.append(
                [image_name, -1, -1, -1, -1, -1, 0, 0, 0, 0]
                if attributes
                else [image_name, 0, 0, -1, -1, -1]
            )
        elif not image_records:
            rows.append([image_name, "None"])
    return rows


def write_evaluation_artifacts(
    output_directory: str,
    state: Mapping[str, Any],
    attributes: bool,
    have_gt: bool = True,
) -> None:
    """Write per-object CSV artifacts applicable to the evaluation mode."""
    output_path = Path(output_directory)
    _write_rows(
        output_path / "detections_data_any_crop.csv",
        ATTRIBUTE_COLUMNS if attributes else COUNT_COLUMNS,
        _detection_rows(state["detections_data_any_crop"], attributes),
    )
    summary_columns = (
        ATTRIBUTE_SUMMARY_COLUMNS if attributes else COUNT_SUMMARY_COLUMNS
    )
    if have_gt:
        _write_rows(
            output_path / "not_found_gt.csv",
            summary_columns,
            _summary_rows(state["not_found_gt"], attributes, False),
        )
    _write_rows(
        output_path / "images_without_detections.csv",
        summary_columns,
        _summary_rows(state["no_predictions"], attributes, True),
    )


def write_keypoint_precision_recall(
    output_directory: str,
    recall: Sequence[Any],
    precision: Sequence[Any],
) -> None:
    """Write the historical keypoint precision-recall CSV artifact."""
    if len(recall) != len(precision):
        raise ValueError("Recall and precision arrays must have equal length.")
    _write_rows(
        Path(output_directory) / "parts_recall_precision.csv",
        ("recall", "precision"),
        list(zip(recall, precision)),
    )


def write_keypoint_summary(
    output_directory: str,
    average_precision: float,
    recall: float,
    precision: float,
) -> None:
    """Write aggregate keypoint-detection metrics as a one-row CSV."""
    _write_rows(
        Path(output_directory) / "keypoint_summary.csv",
        ("mAP", "recall", "precision"),
        ((average_precision, recall, precision),),
    )
