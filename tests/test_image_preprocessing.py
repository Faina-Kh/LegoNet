"""Numeric tests for LegoNet's two preprocessing contracts."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from legonet.my_dataloader import Normalizer, csv_LCCDataset


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
