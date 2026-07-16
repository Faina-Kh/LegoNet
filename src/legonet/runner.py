"""Top-level orchestration for LegoNet training and inference runs."""

import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from legonet import config
from legonet.data_setup import build_data
from legonet.inference import run_inference
from legonet.legoNet_build import model_build
from legonet.model_setup import export_legacy_weights, load_requested_weights
from legonet.training import train_model


DETECTION_NETWORKS = {
    "bbox_detection",
    "per_object_counting",
    "per_object_attributes",
    "per_object_attributes_multibranch",
}


class Tee:
    """Write output to multiple file-like streams."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        """Write and immediately flush data to every stream."""
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        """Flush every configured stream."""
        for stream in self.streams:
            stream.flush()


def unwrap_model(model: Any) -> Any:
    """Return a wrapped model's underlying module when present."""
    return model.module if hasattr(model, "module") else model


def freeze_bn(model: Any) -> None:
    """Switch all BatchNorm2d layers to evaluation mode."""
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def print_args(args: Any, file_path: str) -> None:
    """Print run arguments to the console and a fresh results file."""
    with open(file_path, "w", encoding="utf-8") as results_file:

        def printf(message: str) -> None:
            print(message, end="")
            results_file.write(message)

        printf("=====================================================================\n")
        printf("Run Parameters\n")
        printf("=====================================================================\n")
        for variable_name, variable_value in vars(args).items():
            printf(f"{variable_name}: {variable_value}\n")
        printf(f"experiment path: {config.General.experiment_path}\n")
        printf("=====================================================================\n\n")


def run(args: Any = None) -> Any:
    """Run LegoNet while mirroring console output to the results file."""
    print_args(args, args.txt_results)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(args.txt_results, "a", encoding="utf-8") as results_file:
        sys.stdout = Tee(original_stdout, results_file)
        sys.stderr = Tee(original_stderr, results_file)
        try:
            return _run(args)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run(args: Any = None) -> Any:
    """Build run dependencies and dispatch to training or inference."""
    torch.manual_seed(19860318)
    np.random.seed(19830614)
    random.seed(0)

    data = build_data(args)
    model = model_build(args, data.dataset_train, data.dataset_val)

    if args.load_weights or getattr(args, "load_only_bbox_weights", False):
        load_requested_weights(model, args)
    elif args.save_from_model_file:
        export_legacy_weights(model, args)
        return None

    if args.network_type in DETECTION_NETWORKS and args.freeze_detection:
        model.freeze_detector()

    model = model.to(config.General.device)
    if args.run_script == "Training":
        return train_model(
            args,
            model,
            data.dataset_train,
            data.dataset_val,
            data.sampler,
            data.sampler_val,
            data.dataloader_train,
            data.dataloader_val,
        )

    return run_inference(
        args,
        data.dataset_val,
        data.dataloader_val,
        data.sampler_val,
        model,
    )
