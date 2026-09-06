"""Tests for automatic public-dataset preparation."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from legonet import datasets


def _write_grape_annotations(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.txt").write_text(
        "A.jpg,grapes,1,2,3,4\nA.jpg,grapes,5,6\n", encoding="utf-8"
    )
    (directory / "val.txt").write_text(
        "B.jpg,grapes,1,2,3,4\n", encoding="utf-8"
    )
    (directory / "test.txt").write_text("", encoding="utf-8")
    (directory / "classes.txt").write_text("grapes,0\n", encoding="utf-8")


def test_code_checkout_uses_parent_as_storage(tmp_path: Path) -> None:
    code = tmp_path / "Code"
    (code / "scripts").mkdir(parents=True)
    (code / "scripts" / "run_legonet.py").touch()
    (code / "pyproject.toml").touch()
    assert datasets.default_storage_root(code) == tmp_path.resolve()


def test_named_checkout_retains_repository_storage_default(tmp_path: Path) -> None:
    checkout = tmp_path / "LegoNet2_Clean"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "run_legonet.py").touch()
    (checkout / "pyproject.toml").touch()
    assert datasets.default_storage_root(checkout) == checkout.resolve()


def test_expected_grape_images_are_unique(tmp_path: Path) -> None:
    _write_grape_annotations(tmp_path)
    assert datasets.expected_grape_images(tmp_path) == ("A.jpg", "B.jpg")


def test_roots_setup_does_not_require_bundled_documentation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "Grapevines data"

    assert datasets.seed_dataset_metadata("roots_grapevines", destination) == destination
    assert destination.is_dir()


def test_grapes_downloads_only_missing_jpegs(tmp_path: Path) -> None:
    _write_grape_annotations(tmp_path)
    existing_jpeg = b"\xff\xd8existing\xff\xd9"
    downloaded_jpeg = b"\xff\xd8downloaded\xff\xd9"
    (tmp_path / "A.jpg").write_bytes(existing_jpeg)
    with patch.object(
        datasets, "urlopen", return_value=io.BytesIO(downloaded_jpeg)
    ) as request:
        datasets.download_grapes(tmp_path)

    assert (tmp_path / "A.jpg").read_bytes() == existing_jpeg
    assert (tmp_path / "B.jpg").read_bytes() == downloaded_jpeg
    assert request.call_count == 1
    assert request.call_args.args[0].endswith("/data/B.jpg")
    assert datasets.grape_dataset_complete(tmp_path)


def _roots_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for relative in datasets.ROOTS_REQUIRED_FILES:
            archive.writestr(f"Grapevines data/{relative}", "content")
    return output.getvalue()


def _four_crops_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for dataset, image in (("Dataset 1", "one"), ("Dataset 2", "two")):
            for split in ("Train", "Val", "Test"):
                folder = f"Datasets/{dataset}/sub_{split}"
                archive.writestr(f"{folder}/{split}_TRL.csv", f"{image}.jpg,1.0\n")
                archive.writestr(f"{folder}/{split}_pointsOutput.csv", f"{image}.jpg\n")
                archive.writestr(f"{folder}/{image}.jpg", f"image-{image}")
        archive.writestr("Datasets/Dataset 3/TRL.csv", "three.jpg,3.0\n")
        archive.writestr("Datasets/Dataset 3/pointsOutput.csv", "three.jpg\n")
        archive.writestr("Datasets/Dataset 3/three.jpg", "image-three")
        archive.writestr("Datasets/Dataset 4/CORN/TRL.csv", "four.jpg,4.0\n")
        archive.writestr("Datasets/Dataset 4/CORN/pointsOutput.csv", "four.jpg\n")
        archive.writestr("Datasets/Dataset 4/CORN/four.jpg", "image-four")
    return output.getvalue()


def test_roots_archive_is_verified_and_normalized(tmp_path: Path) -> None:
    contents = _roots_zip()
    checksum = hashlib.md5(contents).hexdigest()  # noqa: S324
    destination = tmp_path / "Grapevines data"
    with (
        patch.object(datasets, "ROOTS_ARCHIVE_MD5", checksum),
        patch.object(datasets, "ROOTS_ARCHIVE_SIZE", len(contents)),
        patch.object(datasets, "urlopen", return_value=io.BytesIO(contents)),
    ):
        datasets.download_roots(destination)

    assert datasets.roots_dataset_complete(destination)
    assert not (destination / "Grapevines data").exists()


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe path"):
        datasets._safe_extract_zip(archive_path, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()


def test_four_crops_download_is_cached_and_subsets_are_selective(
    tmp_path: Path,
) -> None:
    contents = _four_crops_zip()
    checksum = hashlib.md5(contents).hexdigest()  # noqa: S324
    destination = tmp_path / "Datasets" / "Four Crops"
    with (
        patch.object(datasets, "FOUR_CROPS_ARCHIVE_MD5", checksum),
        patch.object(datasets, "FOUR_CROPS_ARCHIVE_SIZE", len(contents)),
        patch.object(datasets, "urlopen", return_value=io.BytesIO(contents)) as request,
    ):
        first = datasets.download_four_crops_subset(destination, "dataset_1")
        second = datasets.download_four_crops_subset(destination, "dataset_2")

        assert datasets.four_crops_subset_complete(destination, "dataset_1")
        assert datasets.four_crops_subset_complete(destination, "dataset_2")
    assert request.call_count == 1
    assert (first / "sub_Train" / "one.jpg").read_text() == "image-one"
    assert (first / "sub_Val").is_dir()
    assert (second / "sub_Val" / "two.jpg").read_text() == "image-two"
    assert (
        tmp_path
        / "downloads"
        / "zenodo-7482146"
        / "Datasets.zip"
    ).is_file()


def test_four_crops_rejects_incomplete_existing_destination(tmp_path: Path) -> None:
    contents = _four_crops_zip()
    checksum = hashlib.md5(contents).hexdigest()  # noqa: S324
    destination = tmp_path / "Datasets" / "Four Crops"
    (destination / "dataset_1").mkdir(parents=True)
    with (
        patch.object(datasets, "FOUR_CROPS_ARCHIVE_MD5", checksum),
        patch.object(datasets, "FOUR_CROPS_ARCHIVE_SIZE", len(contents)),
        patch.object(datasets, "urlopen", return_value=io.BytesIO(contents)),
        pytest.raises(ValueError, match="exists but is incomplete"),
    ):
        datasets.download_four_crops_subset(destination, "dataset_1")


def test_four_crops_install_retries_transient_windows_lock(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Four Crops"
    staged = tmp_path / "staged"
    staged.mkdir()

    with (
        patch.object(
            datasets.os,
            "replace",
            side_effect=[PermissionError("locked"), None],
        ) as replace,
        patch.object(datasets.time, "sleep") as sleep,
    ):
        result = datasets._install_four_crops_subset(
            staged,
            dataset_dir,
            "dataset_1",
        )

    assert result == dataset_dir / "dataset_1"
    assert replace.call_count == 2
    sleep.assert_called_once_with(1)


def test_four_crops_selective_extraction_validates_entire_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Datasets/Dataset 1/Train/TRL.csv", "image.jpg,1\n")
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe path"):
        datasets._extract_four_crops_subset(
            archive_path,
            tmp_path / "dataset_1",
            "dataset_1",
        )
    assert not (tmp_path / "outside.txt").exists()


def test_four_crops_offline_setup_reports_missing_subset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_3 is not installed"):
        datasets.ensure_dataset_available(
            "roots_four_crops",
            tmp_path / "Datasets" / "Four Crops",
            download_missing=False,
            dataset_subset="dataset_3",
        )


def test_missing_dataset_can_be_kept_offline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        datasets.ensure_dataset_available(
            "roots_grapevines", tmp_path / "roots", download_missing=False
        )


def test_verify_command_fails_cleanly_for_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = datasets.main(
        ["verify", "roots_grapevines", "--storage-path", str(tmp_path)]
    )
    assert result == 2
    assert "Dataset setup error" in capsys.readouterr().err
