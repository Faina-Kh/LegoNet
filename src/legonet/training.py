"""Epoch-level training orchestration for LegoNet models."""

import collections
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from legonet import config
from legonet import utils
from legonet.checkpointing import save_epoch_checkpoint
from legonet.eval import per_image_attribute_eval
from legonet.training_evaluation import (
    evaluate_combined_iou_sweep,
    evaluate_detection,
    evaluate_per_object_checkpoint_metrics,
)
from legonet.training_step import LossResult, run_training_step


DETECTION_NETWORKS = {
    "bbox_detection",
    "per_object_counting",
    "per_object_attributes",
    "per_object_attributes_multibranch",
}


@dataclass
class BestMetrics:
    """Best validation metrics observed during a training run."""

    mean_average_precision: float = 0.0
    average_relative_error: float = 100.0
    average_relative_error_epoch: Optional[int] = None
    checkpoint_metric_name: Optional[str] = None
    checkpoint_metric_value: Optional[float] = None
    checkpoint_metric_epoch: Optional[int] = None


def _print_best_error_checkpoint_notice(
    epoch: int,
    previous_error: Optional[float],
    current_error: float,
    metric_name: str = "relative error",
) -> None:
    """Report replacement of the saved best checkpoint by a better metric."""
    previous_text = (
        "none" if previous_error is None else f"{previous_error:.6f}"
    )
    print(
        f"New best validation {metric_name}: {current_error:.6f} "
        f"(previous: {previous_text}) at epoch {epoch}. \n"
        "Replacing the previously saved best-epoch weights file. \n"
    )


def _print_best_training_error(args: Any, best: BestMetrics) -> None:
    """Print the best validation error and epoch selected during training."""
    if args.choose_epoch_by_IoUavg:
        error = best.average_relative_error
        epoch = best.average_relative_error_epoch
        metric_name = "IoU-averaged relative error"
    else:
        error = best.checkpoint_metric_value
        epoch = best.checkpoint_metric_epoch
        metric_name = best.checkpoint_metric_name or "checkpoint metric"

    if epoch is None:
        print("Training completed without a valid validation relative error.")
        return

    print(
        f"Best validation {metric_name}: {error:.6f}, achieved at epoch {epoch}."
    )


def _is_better_checkpoint_error(
    current: Optional[float],
    previous: Optional[float],
) -> bool:
    """Return whether an available checkpoint error improves by decreasing."""
    if current is None:
        return False
    if previous is None:
        return True
    return current < previous


def _epoch_loss_keys(args: Any) -> List[str]:
    """Return the scalar loss components recorded for one configuration."""
    network_type = args.network_type
    estimate_type = config.AttributeEstimation.estimate_type
    if network_type == "bbox_detection":
        return ["classification", "regression"]
    if network_type == "per_image_estimation_keypoints":
        return ["l1_estimation", "maps"]
    if network_type == "per_image_estimation_regression":
        return ["reg_estimation"]
    if network_type == "per_object_counting":
        suffix = ["l1_counting", "maps"] if estimate_type == "withKeyPoints" else ["counting"]
        return ["classification", "regression"] + suffix
    suffix = ["color", "length", "diameter"]
    if estimate_type == "withKeyPoints":
        suffix.insert(1, "maps")
    return ["classification", "regression"] + suffix


def _record_losses(history: Dict[str, List[float]], result: LossResult) -> None:
    """Append available scalar loss values to their epoch histories."""
    for name, values in history.items():
        if name in result.values and result.values[name] != -1:
            values.append(result.values[name])


def _print_step(epoch: int, iteration: int, args: Any, result: LossResult, running: float) -> None:
    """Print the current branch-specific batch loss summary."""
    values = result.values
    if args.network_type == "bbox_detection":
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | Classification loss: "
            f"{values['classification']:1.5f} | Regression loss: {values['regression']:1.5f} | "
            f"Running loss: {running:1.5f}"
        )
    elif args.network_type == "per_image_estimation_keypoints":
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | l1 loss: "
            f"{values['l1_estimation']:1.5f} | maps loss: {values['maps']:1.5f} | "
            f"Running loss: {running:1.5f}"
        )
    elif args.network_type == "per_image_estimation_regression":
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | estimation loss: "
            f"{values['reg_estimation']:1.5f} | Running loss: {running:1.5f}"
        )
    elif args.network_type == "per_object_counting":
        attribute_name = "l1_counting" if "l1_counting" in values else "counting"
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | BBOX_detection loss: "
            f"{values['detection']:1.5f} | Counting loss: {values.get(attribute_name, -1):1.5f}"
        )
    else:
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | BBOX_detection loss: "
            f"{values['detection']:1.5f} | Length loss: {values.get('length', -1):1.5f} | "
            f"Diameter loss: {values.get('diameter', -1):1.5f} | "
            f"Color loss: {values.get('color', -1):1.5f}"
        )


def _print_epoch_summary(epoch: int, history: Dict[str, List[float]]) -> None:
    """Print mean values for every populated epoch loss component."""
    summaries = [
        f"{name} mean {np.mean(values):.5f}"
        for name, values in history.items()
        if values
    ]
    utils.printf("Epoch %d summary: %s\n", epoch, ", ".join(summaries))


def _save_periodic_checkpoint(epoch: int, model: Any) -> None:
    """Save a checkpoint on configured non-evaluation epochs."""
    if (epoch + 1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
        save_epoch_checkpoint(model, epoch)


def _format_optional(value: Optional[float], precision: int = 6) -> str:
    """Format an optional metric for concise evaluation output."""
    return "n/a" if value is None else f"{value:.{precision}f}"


def _evaluate_detection_epoch(
    args: Any,
    epoch: int,
    model: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    best: BestMetrics,
) -> None:
    """Evaluate a detection-only model and update its best checkpoint."""
    if args.dataset_type == "kcsv" and args.kcsv_val is None:
        return
    should_evaluate = args.eval_in_train if args.dataset_type == "kcsv" else args.evaluate_detection
    if not should_evaluate:
        if args.dataset_type == "kcsv":
            _save_periodic_checkpoint(epoch, model)
        return
    metrics = evaluate_detection(dataset_val, dataloader_val, sampler_val, model)
    mean_ap = metrics.mean_average_precision
    precision = metrics.precision
    recall = metrics.recall
    print(f"Current mAP = {mean_ap:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n")
    if args.dataset_type == "kcsv":
        with open(args.txt_results, "a") as results_file:
            results_file.write(
                f"Epoch: {epoch}, mAP = {mean_ap:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n"
            )
    if mean_ap > best.mean_average_precision:
        best.mean_average_precision = mean_ap
        save_epoch_checkpoint(model, epoch, replace_existing=True)


def _evaluate_per_image_attribute_epoch(
    args: Any,
    epoch: int,
    model: Any,
    dataset_val: Any,
    dataloader_val: Any,
    best: BestMetrics,
) -> None:
    """Evaluate a per-image attribute model and update its best checkpoint."""
    if not args.eval_in_train:
        _save_periodic_checkpoint(epoch, model)
        return
    model.eval()
    summary = per_image_attribute_eval.evaluate_checkpoint_metrics(
        dataloader_val, dataset_val, model, args
    )
    metric_value = summary.metric_value
    print(
        f"{summary.metric_name}: {_format_optional(metric_value)} | "
        f"prev_best: {_format_optional(best.checkpoint_metric_value)}\n"
    )
    if _is_better_checkpoint_error(
        metric_value,
        best.checkpoint_metric_value,
    ):
        _print_best_error_checkpoint_notice(
            epoch,
            best.checkpoint_metric_value,
            metric_value,
            metric_name=summary.metric_name,
        )
        best.checkpoint_metric_name = summary.metric_name
        best.checkpoint_metric_value = metric_value
        best.checkpoint_metric_epoch = epoch
        save_epoch_checkpoint(model, epoch, replace_existing=True)


def _evaluate_combined_epoch(
    args: Any,
    epoch: int,
    model: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    best: BestMetrics,
) -> None:
    """Evaluate a combined model and update its selected checkpoint."""
    if not args.eval_in_train:
        _save_periodic_checkpoint(epoch, model)
        return
    if not torch.cuda.is_available():
        print("CUDA not available for combined evaluation")
        return
    if args.choose_epoch_by_IoUavg:
        sweep = evaluate_combined_iou_sweep(
            dataset_val,
            dataloader_val,
            sampler_val,
            model,
            args,
            config.Detection.iou_threshold_list,
        )
        for measurement in sweep.measurements:
            relative_error = _format_optional(measurement.relative_error)
            recall = _format_optional(measurement.recall)
            precision = _format_optional(measurement.precision)
            matched_objects = (
                str(measurement.matched_objects)
                if measurement.matched_objects is not None
                else "n/a"
            )
            print(
                f"IoU {measurement.iou_threshold:.2f}: relative_error={relative_error}, "
                f"matched_objects={matched_objects}, recall={recall}, "
                f"precision={precision}"
            )
        average_error = sweep.average_relative_error
        print(f"average_error={_format_optional(average_error, precision=3)}\n")
        if average_error is not None and average_error < best.average_relative_error:
            _print_best_error_checkpoint_notice(
                epoch,
                best.average_relative_error,
                average_error,
                metric_name="IoU-averaged relative error",
            )
            best.average_relative_error = average_error
            best.average_relative_error_epoch = epoch
            save_epoch_checkpoint(model, epoch, replace_existing=True)
        return
    summary = evaluate_per_object_checkpoint_metrics(
        dataset_val, dataloader_val, sampler_val, model, args
    )
    metric_value = summary.metric_value
    metric_value_text = _format_optional(metric_value)
    one_minus_fvu_text = _format_optional(summary.one_minus_fvu)
    print(
        "Validation results: \n"
        f"{summary.metric_name}: {metric_value_text} | "
        f"related_1-FVU: {one_minus_fvu_text} \n"
    )
    if _is_better_checkpoint_error(
        metric_value,
        best.checkpoint_metric_value,
    ):
        _print_best_error_checkpoint_notice(
            epoch,
            best.checkpoint_metric_value,
            metric_value,
            metric_name=summary.metric_name,
        )
        best.checkpoint_metric_name = summary.metric_name
        best.checkpoint_metric_value = metric_value
        best.checkpoint_metric_epoch = epoch
        save_epoch_checkpoint(model, epoch, replace_existing=True)


def _evaluate_frozen_detector_before_training(
    args: Any,
    model: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
) -> None:
    """Evaluate a frozen per-object detector once before head training."""
    if args.network_type not in (
        "per_object_counting",
        "per_object_attributes",
        "per_object_attributes_multibranch",
    ) or not args.evaluate_detection:
        return

    metrics = evaluate_detection(dataset_val, dataloader_val, sampler_val, model)
    print(
        "Frozen detector validation before head training: "
        f"mAP = {metrics.mean_average_precision:.3f}, "
        f"precision = {metrics.precision:.3f}, recall = {metrics.recall:.3f}\n"
    )


def train_model(
    args: Any,
    model: Any,
    dataset_train: Any,
    dataset_val: Any,
    sampler: Any,
    sampler_val: Any,
    dataloader_train: Any,
    dataloader_val: Any,
) -> None:
    """Train a configured model across epochs and validation checkpoints."""
    model.training = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, eps=0.0001
    )
    running_losses = collections.deque(maxlen=500)
    best = BestMetrics()
    _evaluate_frozen_detector_before_training(
        args,
        model,
        dataset_val,
        dataloader_val,
        sampler_val,
    )
    print(f"Num training images: {len(dataset_train)}")

    for epoch in range(args.epochs):
        model.train()
        model.freeze_bn()
        if args.network_type in DETECTION_NETWORKS:
            model.freeze_detector()
        epoch_losses = []
        component_history = {name: [] for name in _epoch_loss_keys(args)}

        for iteration, data in enumerate(dataloader_train):
            try:
                result = run_training_step(
                    model,
                    optimizer,
                    data,
                    args,
                    sampler.groups[iteration],
                )
            except Exception as error:
                raise RuntimeError(
                    f"Training failed at epoch {epoch}, iteration {iteration}, "
                    f"network type {args.network_type}."
                ) from error
            if result is None:
                continue
            total_value = result.total.item()
            running_losses.append(total_value)
            epoch_losses.append(total_value)
            _record_losses(component_history, result)
            _print_step(epoch, iteration, args, result, float(np.mean(running_losses)))
            gc.collect()
            torch.cuda.empty_cache()

        _print_epoch_summary(epoch, component_history)
        if args.network_type == "bbox_detection":
            _evaluate_detection_epoch(
                args, epoch, model, dataset_val, dataloader_val, sampler_val, best
            )
        elif args.network_type in ("per_image_estimation_keypoints", "per_image_estimation_regression"):
            _evaluate_per_image_attribute_epoch(
                args, epoch, model, dataset_val, dataloader_val, best
            )
        else:
            _evaluate_combined_epoch(
                args, epoch, model, dataset_val, dataloader_val, sampler_val, best
            )
        scheduler.step(np.mean(epoch_losses))

    if args.network_type != "bbox_detection":
        _print_best_training_error(args, best)

    model.eval()
