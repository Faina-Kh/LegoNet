"""Utilities for inspecting and loading current LegoNet checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def list_checkpoint_modules(state_dict: Mapping[str, Any]) -> list[str]:
    """Return sorted top-level module names stored in a state dict."""
    return sorted({key.split(".")[0] for key in state_dict})


def validate_checkpoint_modules(
    state_dict: Mapping[str, Any],
    expected_modules: Sequence[str],
    checkpoint_description: str,
) -> None:
    """Raise when checkpoint modules do not match the requested architecture."""
    actual = set(list_checkpoint_modules(state_dict))
    expected = set(expected_modules)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing modules: {missing}")
    if unexpected:
        details.append(f"unexpected modules: {unexpected}")
    raise ValueError(
        f"{checkpoint_description} architecture does not match the built model "
        f"({'; '.join(details)} \n). Select weights created for the same "
        "network type and estimate type."
    )


def load_submodule_weights(
    model: Any,
    state_dict: Mapping[str, Any],
    submodule_names: Sequence[str],
    strict: bool = True,
    verbose: bool = False,
) -> None:
    """Load matching checkpoint entries into selected model submodules."""
    modules = dict(model.named_modules())
    for submodule_name in submodule_names:
        submodule = modules.get(submodule_name)
        if submodule is None:
            raise ValueError(f"Submodule '{submodule_name}' not found in model")
        prefix = submodule_name + "."
        filtered = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if verbose:
            print(f"\nLoading '{submodule_name}'")
            print(f"Found {len(filtered)} matching parameters")
        if not filtered:
            raise ValueError(f"No weights found for submodule '{submodule_name}'")
        result = submodule.load_state_dict(filtered, strict=strict)
        if verbose:
            print(f"Missing keys: {result.missing_keys}")
            print(f"Unexpected keys: {result.unexpected_keys} \n")


def print_module_names(model: Any) -> None:
    """Print immediate child modules and mark those without checkpoint state."""
    names = [
        name if module.state_dict() else f"{name} (no weights)"
        for name, module in model.named_children()
    ]
    print(names, "\n")
