"""Top-level orchestration for LegoNet training and inference runs."""

import json
import random
import re
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
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

    _TRANSIENT_PERCENTAGE = re.compile(r"^\s*\d+(?:\.\d+)?%\s*$")
    _PROGRESS_PROTOCOL_PREFIX = "__LEGONET_PROGRESS__\t"

    def __init__(
        self,
        *streams: Any,
        suppress_transient_progress_after_first: bool = False,
    ) -> None:
        self.streams = streams
        self.suppress_transient_progress_after_first = (
            suppress_transient_progress_after_first
        )
        self._suppressed_line_break_streams: set[int] = set()

    def write(self, data: str) -> None:
        """Write and immediately flush data to every stream."""
        transient_percentage = bool(
            self._TRANSIENT_PERCENTAGE.fullmatch(data.strip("\r\n"))
        )
        progress_protocol = data.startswith(self._PROGRESS_PROTOCOL_PREFIX)
        for index, stream in enumerate(self.streams):
            suppress_transient = (
                index > 0
                and self.suppress_transient_progress_after_first
                and (transient_percentage or progress_protocol)
            )
            if suppress_transient:
                if not data.endswith(("\n", "\r")):
                    self._suppressed_line_break_streams.add(index)
                continue
            if index in self._suppressed_line_break_streams:
                self._suppressed_line_break_streams.remove(index)
                if data in ("\n", "\r\n"):
                    continue
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


def _json_compatible(value: Any) -> Any:
    """Convert legacy runtime values into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_json_compatible(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _class_configuration(configuration_class: type[Any]) -> dict[str, Any]:
    """Return public data attributes from one legacy configuration class."""
    return {
        name: _json_compatible(value)
        for name, value in sorted(vars(configuration_class).items())
        if not name.startswith("_") and not callable(value)
    }


def _write_run_configuration(args: Any, results_path: Path) -> Path:
    """Write the complete resolved run configuration beside text results."""
    raw_arguments = vars(args)
    invocation_arguments = raw_arguments.get("_invocation_argv", [])
    resolved_arguments = {
        name: _json_compatible(value)
        for name, value in sorted(raw_arguments.items())
        if not name.startswith("_")
    }
    payload = {
        "schema_version": 1,
        "invocation_arguments": _json_compatible(invocation_arguments),
        "resolved_arguments": resolved_arguments,
        "runtime_configuration": {
            "AttributeEstimation": _class_configuration(config.AttributeEstimation),
            "Detect_and_Estimate": _class_configuration(config.Detect_and_Estimate),
            "Detection": _class_configuration(config.Detection),
            "DrawProperties": _class_configuration(config.DrawProperties),
            "General": _class_configuration(config.General),
        },
    }
    configuration_path = results_path.with_name("run_configuration.json")
    configuration_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return configuration_path


def _checkpoint_lines(args: Any) -> list[str]:
    """Return only checkpoint paths relevant to the resolved weights mode."""
    candidates = (
        ("Full checkpoint", getattr(args, "full_model_weights", None)),
        ("Detector checkpoint", getattr(args, "bbox_detection_weights_file", None)),
        ("Estimation-head checkpoint", getattr(args, "per_object_weights_file", None)),
    )
    lines = [f"  {label}: {value}" for label, value in candidates if value]
    return lines or ["  Checkpoints: none"]


def _public_estimate_type(args: Any) -> str:
    """Return the public CLI name instead of the legacy internal identifier."""
    if getattr(args, "network_type", None) == "bbox_detection":
        return "not applicable"
    return {
        "withKeyPoints": "keypoints",
        "reg_fpn_p3_p7_min_sig": "regression",
    }.get(
        getattr(args, "estimate_type", None),
        getattr(args, "estimate_type", "not applicable"),
    )


def format_run_parameters(args: Any, configuration_path: Path) -> str:
    """Return a concise, grouped summary of the resolved run settings."""
    lines = [
        "=====================================================================",
        "Run Parameters",
        "=====================================================================",
        "Run",
        f"  Mode: {getattr(args, 'run_script', 'unknown')}",
        f"  Dataset: {getattr(args, 'dataset_name', 'unknown')}",
        f"  Network: {getattr(args, 'network_type', 'unknown')}",
        f"  Estimate type: {_public_estimate_type(args)}",
        f"  Split: {getattr(args, 'val_set', 'not applicable')}",
        f"  Device: {config.General.device}",
        "",
        "Storage and output",
        f"  Storage root: {getattr(args, 'STORAGE_PATH', 'unknown')}",
        f"  Experiment path: {config.General.experiment_path}",
        f"  Results file: {getattr(args, 'txt_results', 'unknown')}",
        f"  Configuration JSON: {configuration_path}",
        "",
        "Weights",
        f"  Mode: {getattr(args, 'weights_mode', 'unknown')}",
        *_checkpoint_lines(args),
        "",
        "Evaluation and visualization",
        f"  Ground truth available: {getattr(args, 'have_GT', False)}",
        f"  Evaluate detection: {getattr(args, 'evaluate_detection', False)}",
        f"  Draw results: {getattr(args, 'to_draw', False)}",
        "",
        "Runtime",
        f"  Batch size: {getattr(args, 'batch_size', 'unknown')}",
        f"  Data workers: {getattr(args, 'num_workers', 'unknown')}",
    ]
    if getattr(args, "run_script", None) == "Training":
        lines.append(f"  Epochs: {getattr(args, 'epochs', 'unknown')}")
    lines.extend(("=====================================================================", ""))
    return "\n".join(lines) + "\n"


def print_args(args: Any, file_path: str) -> None:
    """Print a concise run summary and save its complete JSON configuration."""
    results_path = Path(file_path)
    configuration_path = _write_run_configuration(args, results_path)
    summary = format_run_parameters(args, configuration_path)
    print(summary, end="")
    results_path.write_text(summary, encoding="utf-8")


def run(args: Any = None) -> Any:
    """Run LegoNet while mirroring console output to the results file."""
    print_args(args, args.txt_results)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(args.txt_results, "a", encoding="utf-8") as results_file:
        sys.stdout = Tee(
            original_stdout,
            results_file,
            suppress_transient_progress_after_first=True,
        )
        sys.stderr = Tee(
            original_stderr,
            results_file,
            suppress_transient_progress_after_first=True,
        )
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
