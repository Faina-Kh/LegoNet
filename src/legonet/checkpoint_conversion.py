"""Convert current LegoNet full-model checkpoints into partial checkpoints."""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from legonet.manage_weights import (
    list_checkpoint_modules,
    validate_checkpoint_modules,
)


DETECTOR_MODULES = ("backbone_1", "find_1", "where")
PER_OBJECT_NETWORKS = (
    "per_object_counting",
    "per_object_attributes",
    "per_object_attributes_multibranch",
)
ESTIMATE_TYPES = ("withKeyPoints", "reg_fpn_p3_p7_min_sig")
_ATTRIBUTE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def estimator_module_names(attribute_names: Sequence[str]) -> list[str]:
    """Return estimator module names for scalar or named-attribute heads."""
    names = list(attribute_names)
    if not names:
        return ["estimator"]
    if len(names) != len(set(names)):
        raise ValueError("Attribute names must be unique.")
    invalid = [name for name in names if not _ATTRIBUTE_NAME_PATTERN.fullmatch(name)]
    if invalid:
        raise ValueError(
            "Attribute names must be valid module-name components; invalid names: "
            f"{invalid}"
        )
    return [f"estimator_{name}" for name in names]


def per_object_module_names(
    network_type: str,
    estimate_type: str,
    attribute_names: Sequence[str],
) -> list[str]:
    """Return the expected modules for a selected per-object architecture."""
    if network_type not in PER_OBJECT_NETWORKS:
        raise ValueError(f"{network_type!r} is not a per-object network.")
    if estimate_type not in ESTIMATE_TYPES:
        raise ValueError(f"Unsupported estimate type: {estimate_type!r}.")

    modules = ["backbone_2"]
    if estimate_type == "withKeyPoints":
        modules.append("find_2")
    modules.extend(estimator_module_names(attribute_names))

    if network_type == "per_object_attributes_multibranch":
        if estimate_type != "withKeyPoints":
            raise ValueError(
                "per_object_attributes_multibranch supports only withKeyPoints."
            )
        modules.extend(["backbone_2_b", "find_2_b"])
    return modules


def _filter_modules(
    state_dict: Mapping[str, Any],
    module_names: Sequence[str],
    prefix_to_strip: str = "",
) -> dict[str, Any]:
    """Extract complete top-level modules and optionally remove a common prefix."""
    extracted = {}
    for key, value in state_dict.items():
        candidate = key
        if prefix_to_strip:
            if not key.startswith(prefix_to_strip):
                continue
            candidate = key[len(prefix_to_strip):]
        if any(
            candidate == module or candidate.startswith(module + ".")
            for module in module_names
        ):
            extracted[candidate] = value
    return extracted


def split_full_state_dict(
    state_dict: Mapping[str, Any],
    network_type: str,
    estimate_type: str | None = None,
    attribute_names: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate and split a current full-model state dictionary."""
    if network_type not in PER_OBJECT_NETWORKS:
        raise ValueError(f"Unsupported network type: {network_type!r}.")
    if estimate_type is None:
        raise ValueError("An estimate type is required for a per-object network.")

    head_modules = per_object_module_names(
        network_type,
        estimate_type,
        attribute_names,
    )
    full_head_modules = list(head_modules)
    if (
        network_type == "per_object_attributes"
        and estimate_type == "reg_fpn_p3_p7_min_sig"
    ):
        # The current attributes model constructs find_2 even though its
        # regression path neither trains nor loads that module.
        full_head_modules.append("find_2")
    validate_checkpoint_modules(
        state_dict,
        ["bbox_detection", *full_head_modules],
        "Full per-object checkpoint",
    )

    detector_state = _filter_modules(
        state_dict,
        DETECTOR_MODULES,
        prefix_to_strip="bbox_detection.",
    )
    head_state = _filter_modules(state_dict, head_modules)
    validate_checkpoint_modules(
        detector_state,
        DETECTOR_MODULES,
        "Extracted detector checkpoint",
    )
    validate_checkpoint_modules(
        head_state,
        head_modules,
        "Extracted per-object checkpoint",
    )
    return detector_state, head_state


def _save_state_dicts_atomically(
    outputs: Sequence[tuple[Path, Mapping[str, Any]]],
    overwrite: bool,
) -> None:
    """Write validated state dictionaries through temporary files."""
    existing = [str(path) for path, _ in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Use --overwrite to replace them."
        )

    for path, _ in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    temporary_paths: list[Path] = []
    try:
        for path, state_dict in outputs:
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            torch.save(dict(state_dict), str(temporary))
            reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
            if set(reloaded) != set(state_dict):
                raise RuntimeError(
                    f"Saved checkpoint verification failed for {path}."
                )
            temporary_paths.append(temporary)
        for temporary, (destination, _) in zip(temporary_paths, outputs):
            os.replace(temporary, destination)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def _output_file_path(
    value: str | Path | None,
    description: str,
    required: bool,
) -> Path | None:
    """Normalize an output filename without treating an empty path as ``.``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError(f"{description} is required.")
        return None

    path = Path(value)
    if path == Path("."):
        if required:
            raise ValueError(f"{description} must include a filename.")
        return None
    if path.exists() and path.is_dir():
        raise ValueError(
            f"{description} must be a file path, not a directory: {path}"
        )
    return path


def convert_full_checkpoint(
    full_weights_file: str | Path,
    detector_output_file: str | Path | None,
    network_type: str,
    estimate_type: str | None = None,
    attribute_names: Sequence[str] = (),
    per_object_output_file: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path | None, Path]:
    """Load, validate, split, and save a current full-model checkpoint."""
    import torch

    source = Path(full_weights_file)
    if not source.is_file():
        raise FileNotFoundError(f"Full weights file does not exist: {source}")

    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Expected a current full checkpoint containing a state dictionary."
        )

    detector_state, head_state = split_full_state_dict(
        state_dict,
        network_type,
        estimate_type,
        attribute_names,
    )
    detector_output = _output_file_path(
        detector_output_file,
        "Detector output file",
        required=False,
    )
    head_output = _output_file_path(
        per_object_output_file,
        "Per-object output file",
        required=True,
    )
    if head_state is None:
        raise RuntimeError("Per-object conversion did not produce head weights.")
    assert head_output is not None
    resolved_source = source.resolve()
    output_paths = [head_output]
    if detector_output is not None:
        output_paths.append(detector_output)
    resolved_outputs = [path.resolve() for path in output_paths]
    if resolved_source in resolved_outputs:
        raise ValueError("An output file cannot replace the full source checkpoint.")
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ValueError("Detector and per-object outputs must use different files.")

    print("Full checkpoint modules:", list_checkpoint_modules(state_dict))
    outputs: list[tuple[Path, Mapping[str, Any]]] = []
    if detector_output is not None:
        print("Detector output modules:", list_checkpoint_modules(detector_state))
        outputs.append((detector_output, detector_state))
    print("Per-object output modules:", list_checkpoint_modules(head_state))
    outputs.append((head_output, head_state))

    _save_state_dicts_atomically(outputs, overwrite)
    return detector_output, head_output
