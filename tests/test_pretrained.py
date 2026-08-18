"""Tests for published pretrained-checkpoint resolution."""

from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from legonet import pretrained


def test_selects_grapes_keypoint_full_checkpoint() -> None:
    checkpoint = pretrained.select_published_checkpoint(
        "grapes", "per_object_counting", "withKeyPoints", "full"
    )
    assert checkpoint.filename == "legonet_full_counting_keypoints_grapes.pt"


def test_selects_roots_direct_regression_checkpoint() -> None:
    checkpoint = pretrained.select_published_checkpoint(
        "roots",
        "per_image_estimation_regression",
        "reg_fpn_p3_p7_min_sig",
        "full",
    )
    assert checkpoint.filename == "legonet_direct_TRL_regression_roots.pt"


def test_rejects_unpublished_configuration() -> None:
    with pytest.raises(ValueError, match="No published pretrained checkpoint"):
        pretrained.select_published_checkpoint(
            "grapes", "per_object_attributes", "withKeyPoints", "full"
        )


def test_download_uses_valid_cached_file(tmp_path: Path) -> None:
    contents = b"published checkpoint"
    checkpoint = pretrained.PublishedCheckpoint(
        filename="model.pt",
        size=len(contents),
        md5=hashlib.md5(contents).hexdigest(),  # noqa: S324
    )
    cached = tmp_path / checkpoint.filename
    cached.write_bytes(contents)

    with patch.object(pretrained, "urlopen") as urlopen:
        resolved = pretrained.download_checkpoint(checkpoint, tmp_path)

    assert resolved == cached.resolve()
    urlopen.assert_not_called()


def test_download_rejects_bad_checksum(tmp_path: Path) -> None:
    checkpoint = pretrained.PublishedCheckpoint(
        filename="model.pt", size=3, md5="0" * 32
    )
    response = __import__("io").BytesIO(b"bad")
    with patch.object(pretrained, "urlopen", return_value=response):
        with pytest.raises(ValueError, match="Checksum verification failed"):
            pretrained.download_checkpoint(checkpoint, tmp_path)
    assert not (tmp_path / checkpoint.filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_message_describes_automatic_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contents = b"model"
    checkpoint = pretrained.PublishedCheckpoint(
        filename="model.pt",
        size=len(contents),
        md5=hashlib.md5(contents).hexdigest(),  # noqa: S324
    )
    response = __import__("io").BytesIO(contents)
    with patch.object(pretrained, "urlopen", return_value=response):
        pretrained.download_checkpoint(checkpoint, tmp_path)

    output = capsys.readouterr().out
    assert "Automatic pretrained weights selected" in output
    assert "No local checkpoint was provided" not in output


def test_explicit_full_checkpoint_takes_precedence(tmp_path: Path) -> None:
    supplied = tmp_path / "mine.pt"
    supplied.write_bytes(b"mine")
    args = Namespace(
        weights_mode="full",
        run_script="Inference",
        STORAGE_PATH=str(tmp_path),
        dataset_name="grapes",
        network_type="per_object_counting",
        estimate_type="withKeyPoints",
        full_weights_file=str(supplied),
        bbox_weights_file=None,
        per_object_weights_file=None,
    )
    with patch.object(pretrained, "download_checkpoint") as download:
        pretrained.resolve_pretrained_weights(args)
    assert args.full_weights_file == str(supplied)
    download.assert_not_called()


def test_auto_inference_downloads_full_checkpoint(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded.pt"
    args = Namespace(
        weights_mode="auto",
        run_script="Inference",
        STORAGE_PATH=str(tmp_path),
        dataset_name="roots",
        network_type="per_object_attributes_multibranch",
        estimate_type="withKeyPoints",
        full_weights_file=None,
        bbox_weights_file=None,
        per_object_weights_file=None,
    )
    with patch.object(
        pretrained, "download_checkpoint", return_value=downloaded
    ) as download:
        pretrained.resolve_pretrained_weights(args)
    assert args.weights_mode == "full"
    assert args.full_weights_file == str(downloaded)
    assert download.call_args.args[0].filename == (
        "legonet_full_attributes_multibranch_roots.pt"
    )
