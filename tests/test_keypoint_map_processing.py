"""Tests for the shared keypoint-map suppression path."""

import torch

from legonet.legos import FindModule


def test_process_keypoint_map_matches_existing_layer_sequence() -> None:
    """The helper must preserve the original smooth-step and NMS calculation."""
    finder = FindModule(
        num_features_in=1,
        num_classes=1,
        task="attribute_estimation",
        Find_for_count=True,
    )
    raw_map = torch.tensor(
        [[[0.0, 0.1, 0.0], [0.2, 1.0, 0.3], [0.0, 0.1, 0.0]]],
        dtype=torch.float32,
    )

    expected = finder.SmoothStepFunction1(
        finder.LocalNMS(finder.SmoothStepFunction(raw_map))
    )

    torch.testing.assert_close(finder.process_keypoint_map(raw_map), expected)
