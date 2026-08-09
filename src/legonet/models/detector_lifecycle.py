"""Shared detector-freezing behavior for per-object models."""

from __future__ import annotations

import torch.nn as nn


class DetectorLifecycleMixin:
    """Provide detector and BatchNorm freezing used by per-object models."""

    def freeze_bn(self) -> None:
        """Put every two-dimensional BatchNorm layer in evaluation mode."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_detector(self) -> None:
        """Disable training and gradients for the embedded detector modules."""
        self.bbox_detection.eval()
        for component in (
            self.bbox_detection.backbone_1,
            self.bbox_detection.find_1,
            self.bbox_detection.where,
        ):
            for parameter in component.parameters():
                parameter.requires_grad = False
