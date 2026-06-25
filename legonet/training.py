"""Epoch-level training orchestration for LegoNet models."""

import collections
import gc
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim

import config
from legonet import utils
from legonet.checkpointing import save_epoch_checkpoint
from legonet.eval import attribute_estimation_eval, detection_eval
from legonet.training_evaluation import (
    evaluate_combined_iou_sweep,
    evaluate_combined_once,
    evaluate_detection,
)
from legonet.training_step import LossResult, run_training_step


DETECTION_NETWORKS = {
    "bbox_detection",
    "both",
    "both_for_roots_2",
    "both_Back2bFind2b",
}


@dataclass
class BestMetrics:
    """Best validation metrics observed during a training run."""

    relative_error: float = 100.0
    mean_average_precision: float = 0.0
    average_relative_error: float = 100.0


def _epoch_loss_keys(args: Any) -> List[str]:
    """Return the scalar loss components recorded for one configuration."""
    network_type = args.network_type
    estimate_type = config.AttributeEstimation.estimate_type
    if network_type == "bbox_detection":
        return ["classification", "regression"]
    if network_type == "counting_lean":
        return ["l1_estimation", "maps"]
    if network_type == "counting_reg":
        return ["reg_estimation"]
    if network_type == "both":
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
    elif args.network_type == "counting_lean":
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | l1 loss: "
            f"{values['l1_estimation']:1.5f} | maps loss: {values['maps']:1.5f} | "
            f"Running loss: {running:1.5f}"
        )
    elif args.network_type == "counting_reg":
        print(
            f"Epoch: {epoch} | Iteration: {iteration} | estimation loss: "
            f"{values['reg_estimation']:1.5f} | Running loss: {running:1.5f}"
        )
    elif args.network_type == "both":
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
        if (
            args.dataset_type == "kcsv"
            and (epoch + 1) % config.General.SAVE_EVERY_N_EPOCHS == 0
        ):
            save_epoch_checkpoint(model, epoch)
        return
    model.eval()
    mean_ap, precision, recall = detection_eval.evaluateMAP_simple(
        dataset_val,
        dataloader_val,
        sampler_val,
        model,
        score_threshold=config.Detection.min_score,
        iou_threshold=config.Detection.iou_threshold,
    )
    print(f"Current mAP = {mean_ap:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n")
    if args.dataset_type == "kcsv":
        with open(args.txt_results, "a") as results_file:
            results_file.write(
                f"Epoch: {epoch}, mAP = {mean_ap:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n"
            )
    if mean_ap > best.mean_average_precision:
        best.mean_average_precision = mean_ap
        save_epoch_checkpoint(model, epoch, replace_existing=True)


def _evaluate_counting_epoch(
    args: Any,
    epoch: int,
    model: Any,
    dataset_val: Any,
    dataloader_val: Any,
    best: BestMetrics,
) -> None:
    """Evaluate a counting-only model and update its best checkpoint."""
    if not args.eval_in_train:
        if (epoch + 1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
            save_epoch_checkpoint(model, epoch)
        return
    model.eval()
    relative_error = attribute_estimation_eval.eval(dataloader_val, dataset_val, model, args)
    print(f"Rel_error: {relative_error:.3f} | prev_best: {best.relative_error:.3f}\n")
    if relative_error < best.relative_error:
        best.relative_error = relative_error
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
        if (epoch + 1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
            save_epoch_checkpoint(model, epoch)
        return
    if args.evaluate_detection:
        metrics = evaluate_detection(dataset_val, dataloader_val, sampler_val, model)
        print(
            f"mAP = {metrics.mean_average_precision:.3f}, precision = {metrics.precision:.3f}, "
            f"recall = {metrics.recall:.3f}"
        )
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
        average_error = sweep.average_relative_error
        if average_error is not None and average_error < best.average_relative_error:
            best.average_relative_error = average_error
            save_epoch_checkpoint(model, epoch, replace_existing=True)
        return
    relative_error = evaluate_combined_once(
        dataset_val, dataloader_val, sampler_val, model, args
    )
    if relative_error is not None and relative_error < best.relative_error:
        best.relative_error = relative_error
        save_epoch_checkpoint(model, epoch, replace_existing=True)


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
        elif args.network_type in ("counting_lean", "counting_reg"):
            _evaluate_counting_epoch(
                args, epoch, model, dataset_val, dataloader_val, best
            )
        else:
            _evaluate_combined_epoch(
                args, epoch, model, dataset_val, dataloader_val, sampler_val, best
            )
        scheduler.step(np.mean(epoch_losses))

    model.eval()
