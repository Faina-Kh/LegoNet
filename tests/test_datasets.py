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

    assert datasets.seed_dataset_metadata("roots", destination) == destination
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


def test_roots_archive_is_verified_and_normalized(tmp_path: Path) -> None:
    contents = _roots_zip()
    checksum = hashlib.md5(contents).hexdigest()  # noqa: S324
    destination = tmp_path / "Grapevines data"
    with (
        patch.object(datasets, "ROOTS_ARCHIVE_MD5", checksum),
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


def test_missing_dataset_can_be_kept_offline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        datasets.ensure_dataset_available(
            "roots", tmp_path / "roots", download_missing=False
        )


def test_verify_command_fails_cleanly_for_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = datasets.main(
        ["verify", "roots", "--storage-path", str(tmp_path)]
    )
    assert result == 2
    assert "Dataset setup error" in capsys.readouterr().err
