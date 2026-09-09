"""Numeric tests for LegoNet's two preprocessing contracts."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from legonet.my_dataloader import Normalizer, UnNormalizer, csv_LCCDataset


def test_loader_applies_contract_channel_order(tmp_path):
    image_path = tmp_path / "pixel.png"
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(image_path)
    dataset = SimpleNamespace(image_path_rgb=lambda _index: str(image_path))

    standard = csv_LCCDataset.load_image(dataset, 0, "imagenet_rgb")
    published = csv_LCCDataset.load_image(dataset, 0, "published_roots")

    np.testing.assert_array_equal(standard, rgb)
    np.testing.assert_array_equal(published, rgb[..., ::-1])


def test_both_contracts_use_historical_imagenet_normalization():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    expected = (
        image.astype(np.float32) / 255.0
        - np.array([[[0.485, 0.456, 0.406]]])
    ) / np.array([[[0.229, 0.224, 0.225]]])

    for contract in ("imagenet_rgb", "published_roots"):
        result = Normalizer(pre_process=contract)({"img": image})["img"]
        np.testing.assert_allclose(result, expected)


def _normalized_chw(channels):
    """Return one ImageNet-normalized three-channel pixel."""
    tensor = torch.tensor(channels).reshape(3, 1, 1) / 255.0
    for channel, mean, std in zip(
        tensor,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ):
        channel.sub_(mean).div_(std)
    return tensor


def test_unnormalizer_preserves_rgb_display_order():
    """Standard RGB tensors remain RGB when converted for display."""
    expected_rgb = torch.tensor([10.0, 20.0, 200.0]).reshape(3, 1, 1) / 255.0

    restored = UnNormalizer(preprocessing_contract="imagenet_rgb")(
        _normalized_chw([10.0, 20.0, 200.0])
    )

    torch.testing.assert_close(restored, expected_rgb)


def test_unnormalizer_converts_published_roots_bgr_to_rgb_for_display():
    """Historical BGR model tensors are reordered only for visualization."""
    expected_rgb = torch.tensor([10.0, 20.0, 200.0]).reshape(3, 1, 1) / 255.0

    restored = UnNormalizer(preprocessing_contract="published_roots")(
        _normalized_chw([200.0, 20.0, 10.0])
    )

    torch.testing.assert_close(restored, expected_rgb)
