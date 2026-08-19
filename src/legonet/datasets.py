"""Download and validate the public datasets used by LegoNet."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen


WGISD_COMMIT = "6910edc5ae3aae8c20062941b1641821f0c30127"
WGISD_SOURCE_URL = "https://github.com/thsant/wgisd/tree/master/data"
WGISD_RAW_ROOT = f"https://raw.githubusercontent.com/thsant/wgisd/{WGISD_COMMIT}/data"

ROOTS_RECORD_URL = "https://zenodo.org/records/8084106"
ROOTS_ARCHIVE_URL = (
    "https://zenodo.org/records/8084106/files/Grapevines%20data.zip?download=1"
)
ROOTS_ARCHIVE_MD5 = "f54e6bc932bba9d8023056bba99a773a"
ROOTS_ARCHIVE_SIZE = 48115398

GRAPE_ANNOTATION_FILES = ("train.txt", "val.txt", "test.txt", "classes.txt")
ROOTS_REQUIRED_FILES = (
    "sub_Train/Train.csv",
    "sub_Train/Train_pointsOutput.csv",
    "sub_Train/Train_Dia_Length_Color.txt",
    "sub_Val/Val.csv",
    "sub_Val/Val_pointsOutput.csv",
    "sub_Val/Val_Dia_Length_Color.txt",
    "sub_Test/Test.csv",
    "sub_Test/Test_pointsOutput.csv",
    "sub_Test/Test_Dia_Length_Color.txt",
)


def source_checkout_root(start: Path | None = None) -> Path | None:
    """Return the repository root when running from a LegoNet source checkout."""
    location = (start or Path(__file__)).resolve()
    for candidate in (location, *location.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "scripts" / "run_legonet.py").is_file()
        ):
            return candidate
    return None


def default_storage_root(start: Path | None = None) -> Path | None:
    """Return the four-folder workspace root for a detectable source checkout.

    A repository cloned as ``Code`` uses its parent, producing sibling
    ``Code``, ``Datasets``, ``ExpResults``, and ``checkpoints`` directories.
    Existing checkouts with another name retain the repository-root default.
    """
    checkout = source_checkout_root(start)
    if checkout is None:
        return None
    return checkout.parent if checkout.name.casefold() == "code" else checkout


def bundled_dataset_resources(dataset_name: str) -> Path:
    """Return the packaged metadata directory for one public dataset."""
    folder = "Embrapa WGISD" if dataset_name == "grapes" else "Grapevines data"
    return Path(__file__).resolve().parent / "resources" / "datasets" / folder


def seed_dataset_metadata(dataset_name: str, dataset_dir: str | Path) -> Path:
    """Copy missing tracked annotations, licenses, and documentation to storage."""
    source = bundled_dataset_resources(dataset_name)
    if not source.is_dir():
        raise ValueError(f"Bundled dataset resources are missing: {source}")
    destination = Path(dataset_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for resource in source.iterdir():
        target = destination / resource.name
        if resource.is_file() and not target.exists():
            shutil.copy2(resource, target)
    return destination


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the publisher's MD5 checksum.
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_grape_images(dataset_dir: str | Path) -> tuple[str, ...]:
    """Return the unique JPEG names referenced by the tracked split files."""
    directory = Path(dataset_dir)
    names: set[str] = set()
    for split_name in GRAPE_ANNOTATION_FILES[:3]:
        split_path = directory / split_name
        if not split_path.is_file():
            raise ValueError(f"Missing LegoNet grape annotation file: {split_path}")
        with split_path.open("r", encoding="utf-8") as split_file:
            for line in split_file:
                image_name = line.partition(",")[0].strip()
                if image_name:
                    if Path(image_name).name != image_name or not image_name.lower().endswith(".jpg"):
                        raise ValueError(
                            f"Unsafe or unsupported WGISD image name: {image_name!r}"
                        )
                    names.add(image_name)
    return tuple(sorted(names))


def grape_dataset_complete(dataset_dir: str | Path) -> bool:
    """Return whether every annotation and referenced WGISD JPEG is present."""
    directory = Path(dataset_dir)
    if not all((directory / name).is_file() for name in GRAPE_ANNOTATION_FILES):
        return False
    try:
        names = expected_grape_images(directory)
    except ValueError:
        return False
    return bool(names) and all(_is_jpeg(directory / name) for name in names)


def _is_jpeg(path: Path) -> bool:
    """Return whether a file has the required JPEG boundary markers."""
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as image_file:
        start = image_file.read(2)
        image_file.seek(-2, os.SEEK_END)
        end = image_file.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def _download_file(url: str, destination: Path) -> None:
    """Download one URL to an atomic destination without retaining partial files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output_file:
            temporary = Path(output_file.name)
            with urlopen(url) as response:  # noqa: S310 - URLs come from fixed HTTPS hosts.
                shutil.copyfileobj(response, output_file)
        os.replace(temporary, destination)
    except Exception as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ValueError(f"Could not download {url}: {error}") from error


def download_grapes(dataset_dir: str | Path) -> Path:
    """Download only the WGISD JPEGs referenced by LegoNet annotations."""
    directory = Path(dataset_dir)
    image_names = expected_grape_images(directory)
    missing = [name for name in image_names if not _is_jpeg(directory / name)]
    if not missing:
        print(f"Using existing grapes dataset: {directory}")
        return directory

    print(f"The grapes dataset is missing {len(missing)} of {len(image_names)} JPEG images.")
    print(f"Downloading only .jpg files from {WGISD_SOURCE_URL}")
    print(f"Pinned WGISD commit: {WGISD_COMMIT}")
    print(f"Destination: {directory}")
    sys.stdout.flush()
    for index, image_name in enumerate(missing, start=1):
        url = f"{WGISD_RAW_ROOT}/{quote(image_name)}"
        _download_file(url, directory / image_name)
        if not _is_jpeg(directory / image_name):
            (directory / image_name).unlink(missing_ok=True)
            raise ValueError(f"Downloaded WGISD file is not a valid JPEG: {image_name}")
        if index == 1 or index % 25 == 0 or index == len(missing):
            print(f"Downloaded WGISD JPEGs: {index}/{len(missing)}")
            sys.stdout.flush()
    print("Grapes dataset download complete; all referenced JPEGs are present.")
    return directory


def roots_dataset_complete(dataset_dir: str | Path) -> bool:
    """Return whether the required roots split and annotation files exist."""
    directory = Path(dataset_dir)
    return all((directory / relative).is_file() for relative in ROOTS_REQUIRED_FILES)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting traversal paths and symbolic links."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as source_zip:
        for member in source_zip.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in roots dataset archive: {member.filename}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"Symbolic link in roots dataset archive: {member.filename}")
            output_path = (destination / Path(*member_path.parts)).resolve()
            if not output_path.is_relative_to(destination_resolved):
                raise ValueError(f"Unsafe path in roots dataset archive: {member.filename}")
        source_zip.extractall(destination)


def _find_roots_content(extracted_dir: Path) -> Path:
    """Find the archive directory containing the three expected split folders."""
    candidates = (extracted_dir, *extracted_dir.rglob("*"))
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / split).is_dir()
            for split in ("sub_Train", "sub_Val", "sub_Test")
        ):
            return candidate
    raise ValueError(
        "The roots archive does not contain sub_Train, sub_Val, and sub_Test."
    )


def download_roots(dataset_dir: str | Path) -> Path:
    """Download, verify, safely extract, and validate the roots dataset."""
    directory = Path(dataset_dir)
    if roots_dataset_complete(directory):
        print(f"Using existing roots dataset: {directory}")
        return directory

    directory.parent.mkdir(parents=True, exist_ok=True)
    size_mib = ROOTS_ARCHIVE_SIZE / (1024 * 1024)
    print(f"The roots dataset is incomplete: {directory}/n")
    print(f"Downloading Grapevines data.zip ({size_mib:.1f} MiB) from {ROOTS_RECORD_URL}")
    print("License: Creative Commons Attribution 4.0 International")
    sys.stdout.flush()
    with tempfile.TemporaryDirectory(
        dir=directory.parent, prefix=".legonet-roots-"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        archive = temporary_dir / "Grapevines data.zip"
        _download_file(ROOTS_ARCHIVE_URL, archive)
        actual_md5 = _md5(archive)
        if actual_md5 != ROOTS_ARCHIVE_MD5:
            raise ValueError(
                "Checksum verification failed for Grapevines data.zip: "
                f"expected {ROOTS_ARCHIVE_MD5}, received {actual_md5}."
            )
        extracted = temporary_dir / "extracted"
        extracted.mkdir()
        _safe_extract_zip(archive, extracted)
        content = _find_roots_content(extracted)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(content, directory, dirs_exist_ok=True)

    if not roots_dataset_complete(directory):
        raise ValueError(
            f"The extracted roots dataset is missing required files in {directory}./n"
        )
    print(f"Roots dataset download complete; checksum and layout verified: {directory}/n")
    return directory


def ensure_dataset_available(
    dataset_name: str,
    dataset_dir: str | Path,
    download_missing: bool = True,
) -> Path:
    """Validate one dataset and optionally download missing public files."""
    directory = seed_dataset_metadata(dataset_name, dataset_dir)
    complete = (
        grape_dataset_complete(directory)
        if dataset_name == "grapes"
        else roots_dataset_complete(directory)
    )
    if complete:
        return directory
    if not download_missing:
        raise ValueError(
            f"The {dataset_name} dataset is incomplete at {directory}. "
            "Allow automatic setup or download it before running LegoNet./n"
        )
    return download_grapes(directory) if dataset_name == "grapes" else download_roots(directory)


def main(argv: list[str] | None = None) -> int:
    """Download or verify a public LegoNet dataset."""
    parser = argparse.ArgumentParser(description="Download public LegoNet datasets.")
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("dataset", choices=("grapes", "roots", "all"))
    parser.add_argument("--storage-path", default=None)
    args = parser.parse_args(argv)
    storage = Path(args.storage_path).expanduser() if args.storage_path else default_storage_root()
    if storage is None:
        parser.error("--storage-path is required outside a LegoNet source checkout.")
    storage.mkdir(parents=True, exist_ok=True)
    selected = ("grapes", "roots") if args.dataset == "all" else (args.dataset,)
    for dataset_name in selected:
        folder = "Embrapa WGISD" if dataset_name == "grapes" else "Grapevines data"
        try:
            ensure_dataset_available(
                dataset_name,
                storage / "Datasets" / folder,
                download_missing=args.action == "download",
            )
        except ValueError as error:
            print(f"Dataset setup error: {error}", file=sys.stderr)
            return 2
        print(f"{dataset_name.title()} dataset verified./n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
