"""Download and validate the public datasets used by LegoNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
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

FOUR_CROPS_RECORD_ID = "7482146"
FOUR_CROPS_RECORD_URL = f"https://zenodo.org/records/{FOUR_CROPS_RECORD_ID}"
FOUR_CROPS_ARCHIVE_URL = (
    f"https://zenodo.org/api/records/{FOUR_CROPS_RECORD_ID}/files/"
    "Datasets.zip/content"
)
FOUR_CROPS_ARCHIVE_NAME = "Datasets.zip"
FOUR_CROPS_ARCHIVE_SIZE = 4625881819
FOUR_CROPS_ARCHIVE_MD5 = "5c1be488f09c4077e5a4b15c435aa964"
FOUR_CROPS_SUBSETS = {
    "dataset_1": "Dataset 1",
    "dataset_2": "Dataset 2",
    "dataset_3": "Dataset 3",
    "dataset_4": "Dataset 4",
}
FOUR_CROPS_INSTALL_MARKER = ".legonet-source.json"

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

DATASET_NAME_ALIASES = {"roots": "roots_grapevines"}


def normalize_dataset_name(dataset_name: str) -> str:
    """Return the canonical dataset identifier accepted by preparation code."""
    return DATASET_NAME_ALIASES.get(dataset_name, dataset_name)


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
    """Return the packaged metadata directory for the grapes dataset."""
    if dataset_name != "grapes":
        raise ValueError(f"No bundled dataset resources exist for {dataset_name!r}.")
    return (
        Path(__file__).resolve().parent
        / "resources"
        / "datasets"
        / "Embrapa WGISD"
    )


def seed_dataset_metadata(dataset_name: str, dataset_dir: str | Path) -> Path:
    """Copy missing packaged grape annotations and licensing to storage."""
    dataset_name = normalize_dataset_name(dataset_name)
    destination = Path(dataset_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if dataset_name in {"roots_grapevines", "roots_four_crops"}:
        return destination
    source = bundled_dataset_resources(dataset_name)
    if not source.is_dir():
        raise ValueError(f"Bundled dataset resources are missing: {source}")
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


def _download_file(
    url: str,
    destination: Path,
    expected_size: int | None = None,
) -> None:
    """Download atomically with a network timeout and visible progress."""
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
            with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed HTTPS hosts.
                downloaded = 0
                last_report = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 2:
                        downloaded_gib = downloaded / (1024**3)
                        if expected_size:
                            percentage = downloaded * 100 / expected_size
                            print(
                                f"Downloaded {downloaded_gib:.2f} GiB "
                                f"of {expected_size / (1024**3):.2f} GiB "
                                f"({percentage:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(f"Downloaded {downloaded_gib:.2f} GiB", flush=True)
                        last_report = now
                if expected_size is not None and downloaded != expected_size:
                    raise ValueError(
                        f"incomplete transfer: expected {expected_size} bytes, "
                        f"received {downloaded}"
                    )
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


def four_crops_subset_complete(
    dataset_dir: str | Path,
    subset: str,
) -> bool:
    """Return whether one selectively extracted Four Crops subset is installed."""
    if subset not in FOUR_CROPS_SUBSETS:
        return False
    subset_dir = Path(dataset_dir) / subset
    marker = subset_dir / FOUR_CROPS_INSTALL_MARKER
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    marker_valid = (
        metadata.get("zenodo_record") == FOUR_CROPS_RECORD_ID
        and metadata.get("archive_md5") == FOUR_CROPS_ARCHIVE_MD5
        and metadata.get("subset") == subset
    )
    if not marker_valid:
        return False
    if subset in {"dataset_1", "dataset_2"}:
        return all(
            (subset_dir / f"sub_{split}" / filename).is_file()
            for split in ("Train", "Val", "Test")
            for filename in (f"{split}_TRL.csv", f"{split}_pointsOutput.csv")
        )
    if subset == "dataset_3":
        return all((subset_dir / filename).is_file() for filename in ("TRL.csv", "pointsOutput.csv"))
    return any(
        directory.is_dir()
        and (directory / "TRL.csv").is_file()
        and (directory / "pointsOutput.csv").is_file()
        for directory in subset_dir.rglob("*")
    )


def four_crops_archive_path(dataset_dir: str | Path) -> Path:
    """Return the shared verified-archive cache path for Four Crops."""
    directory = Path(dataset_dir).resolve()
    if directory.parent.name.casefold() != "datasets":
        raise ValueError(
            "Four Crops destination must be directly below the storage "
            f"Datasets directory: {directory}"
        )
    storage_root = directory.parent.parent
    return (
        storage_root
        / "downloads"
        / f"zenodo-{FOUR_CROPS_RECORD_ID}"
        / FOUR_CROPS_ARCHIVE_NAME
    )


def _validate_zip_member(member: zipfile.ZipInfo, description: str) -> None:
    """Reject unsafe paths and symbolic links in a ZIP member."""
    member_path = PurePosixPath(member.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe path in {description}: {member.filename}")
    if stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError(f"Symbolic link in {description}: {member.filename}")


def _four_crops_member_relative_path(
    member: zipfile.ZipInfo,
    subset: str,
) -> Path | None:
    """Return a member path relative to its selected Dataset N directory."""
    expected = FOUR_CROPS_SUBSETS[subset].casefold()
    parts = PurePosixPath(member.filename).parts
    for index, part in enumerate(parts):
        normalized = part.replace("_", " ").strip().casefold()
        if normalized == expected:
            relative_parts = parts[index + 1 :]
            return Path(*relative_parts) if relative_parts else Path()
    return None


def _extract_four_crops_subset(
    archive: Path,
    destination: Path,
    subset: str,
) -> None:
    """Safely extract only one Dataset N tree into a normalized destination."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    extracted_files = 0
    with zipfile.ZipFile(archive) as source_zip:
        members = source_zip.infolist()
        for member in members:
            _validate_zip_member(member, "Four Crops archive")
        for member in members:
            relative = _four_crops_member_relative_path(member, subset)
            if relative is None or not relative.parts:
                continue
            output = (destination / relative).resolve()
            if not output.is_relative_to(destination_resolved):
                raise ValueError(
                    f"Unsafe selected-subset path in Four Crops archive: {member.filename}"
                )
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with source_zip.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files += 1
    if extracted_files == 0:
        raise ValueError(
            f"The Four Crops archive does not contain {FOUR_CROPS_SUBSETS[subset]}."
        )


def download_four_crops_subset(
    dataset_dir: str | Path,
    subset: str,
) -> Path:
    """Cache the verified archive and install only the requested subset."""
    if subset not in FOUR_CROPS_SUBSETS:
        choices = ", ".join(FOUR_CROPS_SUBSETS)
        raise ValueError(f"Unknown Four Crops subset {subset!r}; choose {choices}.")
    directory = Path(dataset_dir)
    subset_dir = directory / subset
    if four_crops_subset_complete(directory, subset):
        print(f"Using existing Four Crops {subset}: {subset_dir}")
        return subset_dir.resolve()

    archive = four_crops_archive_path(directory)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file() and _md5(archive) != FOUR_CROPS_ARCHIVE_MD5:
        print(f"Discarding cached Four Crops archive with an invalid checksum: {archive}")
        archive.unlink()
    if not archive.is_file():
        size_gib = FOUR_CROPS_ARCHIVE_SIZE / (1024**3)
        print(
            f"Downloading {FOUR_CROPS_ARCHIVE_NAME} ({size_gib:.2f} GiB) "
            f"from {FOUR_CROPS_RECORD_URL}"
        )
        print(
            "Zenodo packages all four subsets in one archive; subset selection "
            "reduces extracted disk usage, not download size."
        )
        print(f"Archive cache: {archive}")
        print("License: Creative Commons Attribution 4.0 International")
        sys.stdout.flush()
        _download_file(
            FOUR_CROPS_ARCHIVE_URL,
            archive,
            expected_size=FOUR_CROPS_ARCHIVE_SIZE,
        )
        actual_md5 = _md5(archive)
        if actual_md5 != FOUR_CROPS_ARCHIVE_MD5:
            archive.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum verification failed for {FOUR_CROPS_ARCHIVE_NAME}: "
                f"expected {FOUR_CROPS_ARCHIVE_MD5}, received {actual_md5}."
            )
    else:
        print(f"Using verified cached Four Crops archive: {archive}")

    if subset_dir.exists():
        raise ValueError(
            f"Four Crops destination exists but is incomplete: {subset_dir}. "
            "Move it aside or remove it before retrying extraction."
        )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=directory, prefix=f".{subset}-extract-"
    ) as temporary_name:
        staged = Path(temporary_name) / subset
        _extract_four_crops_subset(archive, staged, subset)
        marker = {
            "dataset": "roots_four_crops",
            "subset": subset,
            "source_folder": FOUR_CROPS_SUBSETS[subset],
            "zenodo_record": FOUR_CROPS_RECORD_ID,
            "archive": FOUR_CROPS_ARCHIVE_NAME,
            "archive_md5": FOUR_CROPS_ARCHIVE_MD5,
        }
        (staged / FOUR_CROPS_INSTALL_MARKER).write_text(
            json.dumps(marker, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, subset_dir)
    print(f"Four Crops {subset} installed: {subset_dir}")
    return subset_dir.resolve()


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting traversal paths and symbolic links."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as source_zip:
        for member in source_zip.infolist():
            _validate_zip_member(member, "roots dataset archive")
            member_path = PurePosixPath(member.filename)
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
    print(f"The roots dataset is incomplete: {directory}\n")
    print(f"Downloading Grapevines data.zip ({size_mib:.1f} MiB) from {ROOTS_RECORD_URL}")
    print("License: Creative Commons Attribution 4.0 International")
    sys.stdout.flush()
    with tempfile.TemporaryDirectory(
        dir=directory.parent, prefix=".legonet-roots-"
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        archive = temporary_dir / "Grapevines data.zip"
        _download_file(ROOTS_ARCHIVE_URL, archive, expected_size=ROOTS_ARCHIVE_SIZE)
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
            f"The extracted roots dataset is missing required files in {directory}.\n"
        )
    print(f"Roots dataset download complete; checksum and layout verified: {directory}\n")
    return directory


def ensure_dataset_available(
    dataset_name: str,
    dataset_dir: str | Path,
    download_missing: bool = True,
    dataset_subset: str | None = None,
) -> Path:
    """Validate one dataset and optionally download missing public files."""
    dataset_name = normalize_dataset_name(dataset_name)
    directory = seed_dataset_metadata(dataset_name, dataset_dir)
    if dataset_name == "roots_four_crops":
        if dataset_subset is None:
            raise ValueError("A Four Crops dataset subset is required.")
        if four_crops_subset_complete(directory, dataset_subset):
            return (directory / dataset_subset).resolve()
        if not download_missing:
            raise ValueError(
                f"Four Crops {dataset_subset} is not installed at "
                f"{directory / dataset_subset}. Allow automatic setup or "
                "download it before running LegoNet."
            )
        return download_four_crops_subset(directory, dataset_subset)
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
            "Allow automatic setup or download it before running LegoNet.\n"
        )
    return download_grapes(directory) if dataset_name == "grapes" else download_roots(directory)


def main(argv: list[str] | None = None) -> int:
    """Download or verify a public LegoNet dataset."""
    parser = argparse.ArgumentParser(description="Download public LegoNet datasets.")
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument(
        "dataset",
        choices=("grapes", "roots_grapevines", "roots_four_crops", "roots", "all"),
    )
    parser.add_argument("--storage-path", default=None)
    parser.add_argument(
        "--dataset-subset",
        choices=tuple(FOUR_CROPS_SUBSETS),
        default=None,
    )
    args = parser.parse_args(argv)
    storage = Path(args.storage_path).expanduser() if args.storage_path else default_storage_root()
    if storage is None:
        parser.error("--storage-path is required outside a LegoNet source checkout.")
    storage.mkdir(parents=True, exist_ok=True)
    selected = (
        ("grapes", "roots_grapevines")
        if args.dataset == "all"
        else (normalize_dataset_name(args.dataset),)
    )
    if args.dataset == "roots_four_crops" and args.dataset_subset is None:
        parser.error("--dataset-subset is required for roots_four_crops.")
    for dataset_name in selected:
        folder = {
            "grapes": "Embrapa WGISD",
            "roots_grapevines": "Grapevines data",
            "roots_four_crops": "Four Crops",
        }[dataset_name]
        try:
            ensure_dataset_available(
                dataset_name,
                storage / "Datasets" / folder,
                download_missing=args.action == "download",
                dataset_subset=args.dataset_subset,
            )
        except ValueError as error:
            print(f"Dataset setup error: {error}", file=sys.stderr)
            return 2
        print(f"{dataset_name.title()} dataset verified.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
