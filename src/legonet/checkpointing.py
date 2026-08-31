"""Safe checkpoint persistence for LegoNet training runs."""

from pathlib import Path
from typing import Any, Optional

import torch

from legonet import config


CHECKPOINT_GLOB = "legonet_epoch=*.pt"


def save_epoch_checkpoint(
    model: Any,
    epoch: int,
    replace_existing: bool = False,
    weights_dir: Optional[str] = None,
) -> Path:
    """Save an epoch checkpoint and optionally remove older LegoNet checkpoints.

    Unrelated files are never removed. Existing checkpoints are cleaned up only
    after the new checkpoint has been saved successfully.
    """
    checkpoint_directory = Path(weights_dir or config.General.weights_dir)
    checkpoint_path = checkpoint_directory / f"legonet_epoch={epoch}.pt"
    torch.save(model.state_dict(), str(checkpoint_path))

    if replace_existing:
        for previous_checkpoint in checkpoint_directory.glob(CHECKPOINT_GLOB):
            if previous_checkpoint != checkpoint_path and previous_checkpoint.is_file():
                previous_checkpoint.unlink()

    print(f"Saved epoch {epoch} checkpoint: {checkpoint_path}")
    return checkpoint_path
