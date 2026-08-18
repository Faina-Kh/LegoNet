"""Resolve and download published LegoNet pretrained checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


ZENODO_RECORD_ID = "21966953"
ZENODO_DOI = "10.5281/zenodo.21966953"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"


@dataclass(frozen=True)
class PublishedCheckpoint:
    """Metadata required to retrieve and validate one published checkpoint."""

    filename: str
    size: int
    md5: str

    @property
    def download_url(self) -> str:
        """Return the stable Zenodo file download URL."""
        return f"{ZENODO_RECORD_URL}/files/{quote(self.filename)}?download=1"


def _checkpoint(filename: str, size: int, md5: str) -> PublishedCheckpoint:
    return PublishedCheckpoint(filename=filename, size=size, md5=md5)


CHECKPOINTS = {
    "bbox_grapes": _checkpoint("legonet_bbox_grapes.pt", 145650358, "ebd63a96df0cd4e39e3d70fe4d3f1601"),
    "bbox_roots": _checkpoint("legonet_bbox_roots.pt", 145645072, "b2de6847c6e50d6026b80968ec9be52b"),
    "counting_keypoints_grapes": _checkpoint("legonet_partial_counting_keypoints_grapes.pt", 135894882, "c95a3684ba321c23b0eb348b516d4406"),
    "counting_regression_grapes": _checkpoint("legonet_partial_counting_regression_grapes.pt", 131243834, "f2faab2f869bf1a1acd014bdd1b8923f"),
    "full_counting_keypoints_grapes": _checkpoint("legonet_full_counting_keypoints_grapes.pt", 281568386, "af0a89b5371d5b1ea58389bf39b9ecd4"),
    "full_counting_regression_grapes": _checkpoint("legonet_full_counting_regression_grapes.pt", 276910122, "e28aadac49fe027b4441564c65c5c688"),
    "attributes_keypoints_roots": _checkpoint("legonet_partial_attributes_keypoints_roots.pt", 138305610, "4982c1004466612a7cd2c58418203ca8"),
    "attributes_regression_roots": _checkpoint("legonet_partial_attributes_regression_roots.pt", 141054362, "24c66522ba0efbc8cf5b93446e613db5"),
    "attributes_multibranch_roots": _checkpoint("legonet_partial_attributes_multibranch_roots.pt", 274237221, "1ed96a108fb64ffed0f044ffd3ea26a5"),
    "full_attributes_keypoints_roots": _checkpoint("legonet_full_attributes_keypoints_roots.pt", 283982821, "dea53894ba64d407b645609860490679"),
    "full_attributes_regression_roots": _checkpoint("legonet_full_attributes_regression_roots.pt", 286745356, "4d9c168757d4a5fc8f100571eeb6d08c"),
    "full_attributes_multibranch_roots": _checkpoint("legonet_full_attributes_multibranch_roots.pt", 419951935, "b2dad2d63cafa451ab31d9d879c74303"),
    "direct_trl_keypoints_roots": _checkpoint("legonet_direct_TRL_keypoints_roots.pt", 135888466, "ec846449ce1e45df3883b67b626ee269"),
    "direct_trl_regression_roots": _checkpoint("legonet_direct_TRL_regression_roots.pt", 131229430, "dc69814f45a31937ec78fe2752fd69f1"),
}


def checkpoint_cache_dir(storage_path: str | Path) -> Path:
    """Return the record-specific checkpoint cache below the storage root."""
    return Path(storage_path) / "checkpoints" / f"zenodo-{ZENODO_RECORD_ID}"


def select_published_checkpoint(
    dataset_name: str,
    network_type: str,
    estimate_type: str,
    component: str = "full",
) -> PublishedCheckpoint:
    """Select the published checkpoint for one supported model component."""
    estimate = "keypoints" if estimate_type == "withKeyPoints" else "regression"
    if component == "bbox":
        key = f"bbox_{dataset_name}"
    elif network_type == "bbox_detection":
        key = f"bbox_{dataset_name}"
    elif network_type == "per_object_counting":
        prefix = "full_counting" if component == "full" else "counting"
        key = f"{prefix}_{estimate}_{dataset_name}"
    elif network_type == "per_object_attributes":
        prefix = "full_attributes" if component == "full" else "attributes"
        key = f"{prefix}_{estimate}_{dataset_name}"
    elif network_type == "per_object_attributes_multibranch":
        prefix = "full_attributes" if component == "full" else "attributes"
        key = f"{prefix}_multibranch_{dataset_name}"
    elif network_type in ("per_image_estimation_keypoints", "per_image_estimation_regression"):
        key = f"direct_trl_{estimate}_{dataset_name}"
    else:
        key = ""

    try:
        return CHECKPOINTS[key]
    except KeyError as error:
        raise ValueError(
            "No published pretrained checkpoint is available for "
            f"dataset={dataset_name!r}, network={network_type!r}, "
            f"estimate={estimate_type!r}, component={component!r}. "
            "Provide a local checkpoint path or use --weights-mode none."
        ) from error


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify Zenodo's published checksum.
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    checkpoint: PublishedCheckpoint,
    cache_dir: str | Path,
) -> Path:
    """Download, verify, and atomically cache one published checkpoint."""
    destination_dir = Path(cache_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / checkpoint.filename
    if destination.is_file() and _md5(destination) == checkpoint.md5:
        print(f"Using cached pretrained checkpoint: {destination}")
        return destination.resolve()

    if destination.exists():
        destination.unlink()
    size_mib = checkpoint.size / (1024 * 1024)
    print("No local checkpoint was provided.")
    print(f"Downloading pretrained checkpoint {checkpoint.filename} ({size_mib:.1f} MiB)")
    print(f"  from {ZENODO_RECORD_URL}")
    print(f"  to {destination}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_dir,
            prefix=f".{checkpoint.filename}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(checkpoint.download_url) as response:  # noqa: S310 - fixed HTTPS host.
                shutil.copyfileobj(response, temporary_file)
        actual_md5 = _md5(temporary_path)
        if actual_md5 != checkpoint.md5:
            raise ValueError(
                f"Checksum verification failed for {checkpoint.filename}: "
                f"expected {checkpoint.md5}, received {actual_md5}."
            )
        os.replace(temporary_path, destination)
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(error, ValueError):
            raise
        raise ValueError(
            f"Could not download {checkpoint.filename} from {ZENODO_RECORD_URL}: "
            f"{error}"
        ) from error

    print(f"Download complete; checksum verified: {destination}")
    sys.stdout.flush()
    return destination.resolve()


def resolve_pretrained_weights(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing requested checkpoint paths from the published Zenodo record."""
    if args.weights_mode == "none":
        return args
    if args.weights_mode == "auto":
        args.weights_mode = (
            "detector_only"
            if args.run_script == "Training"
            and args.network_type
            in ("per_object_counting", "per_object_attributes", "per_object_attributes_multibranch")
            else "full"
        )

    if args.weights_mode == "partial" and args.network_type in (
        "per_image_estimation_keypoints",
        "per_image_estimation_regression",
    ):
        raise ValueError(
            "Per-image estimation networks require --weights-mode full, auto, or none."
        )

    cache_dir = checkpoint_cache_dir(args.STORAGE_PATH)
    if args.weights_mode == "full" and not args.full_weights_file:
        checkpoint = select_published_checkpoint(
            args.dataset_name, args.network_type, args.estimate_type, "full"
        )
        args.full_weights_file = str(download_checkpoint(checkpoint, cache_dir))
    if args.weights_mode in ("partial", "detector_only") and not args.bbox_weights_file:
        checkpoint = select_published_checkpoint(
            args.dataset_name, args.network_type, args.estimate_type, "bbox"
        )
        args.bbox_weights_file = str(download_checkpoint(checkpoint, cache_dir))
    if args.weights_mode == "partial" and not args.per_object_weights_file:
        checkpoint = select_published_checkpoint(
            args.dataset_name, args.network_type, args.estimate_type, "head"
        )
        args.per_object_weights_file = str(download_checkpoint(checkpoint, cache_dir))
    return args
