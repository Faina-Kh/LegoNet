"""Regression tests for the public RGB/ImageNet preprocessing contract."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from legonet.my_dataloader import KCSVDataset, Normalizer, csv_LCCDataset


def test_public_loaders_preserve_pil_rgb_for_torch_preprocessing(tmp_path):
    """Both public PIL loaders keep RGB channel order in torch mode."""
    image_path = tmp_path / "pixel.png"
    pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    Image.fromarray(pixel, mode="RGB").save(image_path)

    lcc_dataset = SimpleNamespace(
        image_path_rgb=lambda _index: str(image_path),
    )
    kcsv_dataset = SimpleNamespace(
        base_dir=str(tmp_path),
        img_info=[{"name": image_path.name}],
        image_ids=[0],
    )

    lcc_image = csv_LCCDataset.load_image(lcc_dataset, 0, "torch_like")
    kcsv_image = KCSVDataset.load_image(kcsv_dataset, 0, "torch_like")

    np.testing.assert_array_equal(lcc_image, pixel)
    np.testing.assert_array_equal(kcsv_image, pixel)


def test_torch_preprocessing_scales_and_normalizes_rgb_channels():
    """Torch mode implements the documented ImageNet RGB normalization."""
    pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)

    result = Normalizer(pre_process="torch_like")({"img": pixel})["img"]
    expected = (
        pixel.astype(np.float32) / 255.0
        - np.array([[[0.485, 0.456, 0.406]]])
    ) / np.array([[[0.229, 0.224, 0.225]]])

    np.testing.assert_allclose(result, expected)
