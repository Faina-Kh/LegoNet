"""Regression tests for per-image attribute collation."""

import numpy as np
import torch

from legonet.my_dataloader import LCC_collater


def _sample(attribute_value: float) -> dict:
    """Return one transformed per-image sample with keypoint maps."""
    return {
        "img": np.zeros((32, 32, 3), dtype=np.float32),
        "annot": [
            np.asarray([attribute_value, 0.0], dtype=np.float64),
            *[np.zeros((4, 4), dtype=np.float64) for _ in range(5)],
        ],
    }


def test_lcc_collater_keeps_only_scalar_attribute_target() -> None:
    """The class index must not be broadcast as a regression target."""
    batch = LCC_collater([_sample(123.5)])

    assert batch["annot"][0].shape == torch.Size([1, 1])
    torch.testing.assert_close(
        batch["annot"][0],
        torch.tensor([[123.5]], dtype=torch.float32),
    )


def test_lcc_collater_preserves_five_keypoint_maps() -> None:
    """Correcting the scalar target must not change the map targets."""
    batch = LCC_collater([_sample(123.5)])

    assert len(batch["annot"]) == 6
    for target_map in batch["annot"][1:]:
        assert target_map.shape == torch.Size([1, 4, 4])
