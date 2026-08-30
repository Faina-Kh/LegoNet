"""Reporting, artifact export, and legacy return values for evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from legonet.eval.KP_detection_eval import calc_points_recall_precision_ap
from legonet.eval.detection_eval import plot_PR_curve
from legonet.eval.metric_aggregation import PostLoopMetrics
from legonet.eval.per_image_attribute_metrics import compute_per_image_attribute_metrics
from legonet.eval.reporting import (
    SEPARATOR,
    format_attribute_summary,
    format_counting_per_image,
    format_counting_summary,
    format_keypoint_summary,
    format_matching_diagnostics,
    format_prediction_aggregates,
    format_roots_per_image,
    write_evaluation_artifacts,
    write_keypoint_precision_recall,
    write_keypoint_summary,
)


def finalize_evaluation(
    state: Mapping[str, Any],
    metrics: PostLoopMetrics,
    *,
    attributes: bool,
    have_gt: bool,
    predict_empty_image: bool,
    verbose: bool,
    print_to_files: bool,
    detection_metrics: Sequence[Any] | None,
    evaluate_points: bool,
    files_path: str,
    text_results_path: str | None = None,
) -> list[float]:
    """Report final metrics, write requested artifacts, and build legacy output."""
    _print_availability(metrics, attributes=attributes, have_gt=have_gt, verbose=verbose)
    report_summary = (
        verbose
        or print_to_files
        or os.environ.get("LEGONET_STREAMLIT_SUMMARIES") == "1"
    )
    if report_summary:
        _emit_summary(
            _format_metric_summary(
                metrics,
                attributes=attributes,
                have_gt=have_gt,
            ),
            text_results_path,
        )
    if verbose:
        _print_verbose_details(
            state,
            attributes=attributes,
            have_gt=have_gt,
            predict_empty_image=predict_empty_image,
            detection_metrics=detection_metrics,
        )

    if print_to_files:
        write_evaluation_artifacts(
            files_path,
            state,
            attributes=attributes,
            have_gt=have_gt,
        )

    print()
    if evaluate_points:
        _finalize_keypoints(
            state,
            files_path=files_path,
            report_summary=report_summary,
            text_results_path=text_results_path,
        )

    return build_legacy_result(
        state,
        metrics,
        attributes=attributes,
        detection_metrics=detection_metrics,
    )


def build_legacy_result(
    state: Mapping[str, Any],
    metrics: PostLoopMetrics,
    *,
    attributes: bool,
    detection_metrics: Sequence[Any] | None,
) -> list[float]:
    """Construct the historical list returned to training and inference callers."""
    if attributes:
        return [metrics.length_relative_error]

    reported_recall = (
        detection_metrics[2]
        if detection_metrics is not None
        else state["found_orig_objects"] / state["gt_objects_withGTpoints"]
    )
    reported_precision = (
        detection_metrics[1]
        if detection_metrics is not None
        else metrics.precision_detection
    )
    return [
        metrics.count_relative_error,
        state["gt_objects_withGTpoints"],
        state["found_orig_objects"],
        reported_recall,
        reported_precision,
        metrics.count_mae,
        metrics.count_agreement,
        metrics.count_mse,
        1 - metrics.count_fvu,
    ]


def _print_availability(
    metrics: PostLoopMetrics,
    *,
    attributes: bool,
    have_gt: bool,
    verbose: bool,
) -> None:
    if not metrics.had_predictions and verbose:
        print("There are no images with predicted boxes")
    if not metrics.has_original_gt and (not attributes or have_gt):
        print("No gt boxes for any image")


def _format_metric_summary(
    metrics: PostLoopMetrics,
    *,
    attributes: bool,
    have_gt: bool,
) -> str:
    """Format the aggregate summary shared by console and text-file output."""
    if not attributes:
        return format_counting_summary(
            metrics.count_mae,
            metrics.count_agreement,
            metrics.count_mse,
            metrics.count_relative_error,
            1 - metrics.count_fvu,
            metrics.crop_count_metrics,
        )
    elif have_gt:
        return format_attribute_summary(
            metrics.length_mae,
            metrics.length_mse,
            metrics.length_relative_error,
            1 - metrics.length_fvu,
            metrics.diameter_mae,
            metrics.diameter_mse,
            metrics.diameter_relative_error,
            1 - metrics.diameter_fvu,
            metrics.color_metrics,
        )
    return ""


def _stdout_mirrors(path: Path) -> bool:
    """Return whether the current stdout tee already writes to ``path``."""
    streams = getattr(sys.stdout, "streams", ())
    target = os.path.abspath(os.fspath(path))
    return any(
        getattr(stream, "name", None)
        and os.path.abspath(os.fspath(stream.name)) == target
        for stream in streams
    )


def _emit_summary(summary: str, text_results_path: str | None) -> None:
    """Print a summary and explicitly persist it when stdout is not file-backed."""
    if not summary:
        return
    print(summary, end="")
    if not text_results_path:
        return
    results_path = Path(text_results_path)
    if _stdout_mirrors(results_path):
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as results_file:
        results_file.write(summary)


def _print_verbose_details(
    state: Mapping[str, Any],
    *,
    attributes: bool,
    have_gt: bool,
    predict_empty_image: bool,
    detection_metrics: Sequence[Any] | None,
) -> None:
    """Print matching diagnostics and per-image rows for interactive inspection."""
    if have_gt:
        _print_detection_diagnostics(
            state,
            predict_empty_image=predict_empty_image,
            detection_metrics=detection_metrics,
        )
        print(f"{SEPARATOR}\n")

    if not attributes and have_gt:
        print(
            format_counting_per_image(
                state["per_im_gt_avg_dict"], state["per_im_pred_dict"]
            ),
            end="",
        )
    elif attributes and have_gt:
        _print_attribute_per_image(state)
    elif not have_gt:
        print(f"{SEPARATOR}\n")
        print(
            format_prediction_aggregates(
                state["per_im_pred_dict"],
                trl_predictions=(
                    state["TRL_per_im_pred_dict"] if attributes else None
                ),
                diameter_predictions=(
                    state["dia_per_im_pred_dict"] if attributes else None
                ),
                images_without_detections=tuple(state["no_predictions"]),
            ),
            end="",
        )


def _print_detection_diagnostics(
    state: Mapping[str, Any],
    *,
    predict_empty_image: bool,
    detection_metrics: Sequence[Any] | None,
) -> None:
    if predict_empty_image and detection_metrics is not None and len(detection_metrics) > 3:
        diagnostics = detection_metrics[3]
        scope = "all images, including empty images"
        gt_objects = diagnostics.ground_truth_objects
        matches = diagnostics.matched_objects
        false_positives = diagnostics.false_positives
    else:
        scope = "attribute-annotated images only"
        gt_objects = state["gt_objects_withGTpoints"]
        matches = state["found_orig_objects"]
        false_positives = state["FP"]

    print(
        format_matching_diagnostics(scope, gt_objects, matches, false_positives),
        end="",
    )


def _print_attribute_per_image(state: Mapping[str, Any]) -> None:
    image_names = list(state["per_im_pred_dict"])
    per_image_metrics = compute_per_image_attribute_metrics(
        [state["TRL_per_im_gt_sum_dict"][name] for name in image_names],
        [state["TRL_per_im_pred_dict"][name] for name in image_names],
        [state["dia_per_im_gt_avg_dict"][name] for name in image_names],
        [state["dia_per_im_pred_dict"][name] for name in image_names],
        [state["per_im_gt_avg_dict"][name] for name in image_names],
        [state["per_im_pred_dict"][name] for name in image_names],
    )
    print(
        format_roots_per_image(
            state["TRL_per_im_gt_sum_dict"],
            state["TRL_per_im_pred_dict"],
            state["dia_per_im_gt_avg_dict"],
            state["dia_per_im_pred_dict"],
            state["per_im_gt_avg_dict"],
            state["per_im_pred_dict"],
            per_image_metrics,
        ),
        end="",
    )


def _finalize_keypoints(
    state: Mapping[str, Any],
    *,
    files_path: str,
    report_summary: bool,
    text_results_path: str | None,
) -> None:
    recall, precision, average_precision = calc_points_recall_precision_ap(
        state["T"], state["P"]
    )
    if report_summary:
        _emit_summary(
            format_keypoint_summary(average_precision),
            text_results_path,
        )
    plot_PR_curve(
        recall,
        precision,
        average_precision,
        save_path=files_path,
        plots_name="Points_PR_curve.png",
    )
    write_keypoint_precision_recall(files_path, recall, precision)
    write_keypoint_summary(
        files_path, average_precision, recall[-1], precision[-1]
    )
    _write_keypoint_protocol_comparison(state, files_path)


def _write_keypoint_protocol_comparison(
    state: Mapping[str, Any], files_path: str
) -> None:
    """Write opt-in alternative keypoint-candidate protocol diagnostics."""
    comparison = state.get("keypoint_protocol_comparison", {})
    if not comparison:
        return
    rows = []
    for protocol_name, values in comparison.items():
        recall, precision, average_precision = calc_points_recall_precision_ap(
            values["T"], values["P"]
        )
        rows.append(
            (
                protocol_name,
                average_precision,
                recall[-1],
                precision[-1],
                sum(values["T"]),
                len(precision),
            )
        )
        plot_PR_curve(
            recall,
            precision,
            average_precision,
            save_path=files_path,
            plots_name=f"Points_PR_curve_{protocol_name}.png",
        )
    output_path = Path(files_path) / "keypoint_protocol_comparison.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        output_file.write(
            "protocol,mAP,max_recall,final_precision,ground_truth_points,candidates\n"
        )
        for row in rows:
            output_file.write(",".join(str(value) for value in row) + "\n")
