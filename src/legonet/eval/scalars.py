"""Scalar conversion helpers for evaluation code."""

from typing import Any

import numpy as np


def first_scalar(value: Any) -> Any:
    """Return the first array-like annotation value as a Python scalar."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)[0].item()
