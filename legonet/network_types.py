"""Canonical model-variant names and compatibility aliases."""

from __future__ import annotations

import warnings


PER_IMAGE_ESTIMATION_KEYPOINTS = "per_image_estimation_keypoints"
PER_IMAGE_ESTIMATION_REGRESSION = "per_image_estimation_regression"

LEGACY_NETWORK_TYPE_ALIASES = {
    "counting_lean": PER_IMAGE_ESTIMATION_KEYPOINTS,
    "counting_reg": PER_IMAGE_ESTIMATION_REGRESSION,
}


def canonicalize_network_type(network_type: str, *, warn: bool = True) -> str:
    """Return the canonical model-variant name for a legacy or current name."""
    canonical = LEGACY_NETWORK_TYPE_ALIASES.get(network_type, network_type)
    if warn and canonical != network_type:
        warnings.warn(
            f"Network type {network_type!r} is deprecated; use {canonical!r} instead.",
            FutureWarning,
            stacklevel=2,
        )
    return canonical
