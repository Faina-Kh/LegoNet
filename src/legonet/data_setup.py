"""Dataset, sampler, and dataloader construction for LegoNet runs."""

from dataclasses import dataclass
from typing import Any, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from legonet.my_dataloader import (
    AspectRatioBasedSampler,
    Augmenter,
    CocoDataset,
    KCSVDataset,
    LCC_collater,
    Normalizer,
    Resizer,
    collater,
    csv_LCCDataset,
    kcsv_collater,
)


@dataclass
class DataBundle:
    """Datasets and loader infrastructure needed by a runner invocation."""

    dataset_train: Any
    dataset_val: Any
    sampler: Any
    sampler_val: Any
    dataloader_train: Any
    dataloader_val: Any


def _build_coco_datasets(args: Any) -> Tuple[Any, Any]:
    """Build the current COCO training and validation datasets."""
    dataset_train = CocoDataset(
        args.dataset_path,
        set_name="train",
        transform=transforms.Compose(
            [Normalizer(), Augmenter(), Resizer(ann_type="bbox")]
        ),
    )
    dataset_val = CocoDataset(
        args.dataset_path,
        set_name="val",
        transform=transforms.Compose([Normalizer(), Resizer(ann_type="bbox")]),
    )
    args.collater = collater
    return dataset_train, dataset_val


def _build_kcsv_datasets(args: Any) -> Tuple[Any, Any]:
    """Build KCSV or roots-JSON datasets for the requested run mode."""
    if args.run_script == "Training":
        if args.dataset_type == "kcsv":
            dataset_train = KCSVDataset(
                input_file=args.kcsv_train,
                class_list=args.kcsv_classes,
                pre_process=args.pre_process,
                transform=transforms.Compose(
                    [
                        Normalizer(pre_process=args.pre_process),
                        Resizer(min_side=800, max_side=1333),
                    ]
                ),
            )
        else:
            dataset_train = KCSVDataset(
                input_file=args.train_json_file,
                transform=transforms.Compose(
                    [
                        Normalizer(pre_process=args.pre_process),
                        Resizer(min_side=800, max_side=1333),
                    ]
                ),
                dataset_type="roots_json",
            )
    else:
        dataset_train = None

    if args.dataset_type == "kcsv":
        dataset_val = KCSVDataset(
            input_file=args.val_file,
            class_list=args.kcsv_classes,
            base_dir=getattr(args, "base_dir", None),
            have_GT=getattr(args, "have_GT", True),
            pre_process=args.pre_process,
            transform=transforms.Compose(
                [
                    Normalizer(pre_process=args.pre_process),
                    Resizer(min_side=800, max_side=1333),
                ]
            ),
        )
    else:
        dataset_val = KCSVDataset(
            input_file=args.val_json_file,
            transform=transforms.Compose(
                [
                    Normalizer(pre_process=args.pre_process),
                    Resizer(min_side=800, max_side=1333),
                ]
            ),
            dataset_type="roots_json",
            base_dir=args.base_dir,
            have_GT=args.have_GT,
        )

    args.collater = kcsv_collater
    return dataset_train, dataset_val


def _build_lcc_datasets(args: Any) -> Tuple[Any, Any]:
    """Build per-image roots attribute datasets for the requested run mode."""
    if args.run_script == "Training":
        dataset_train = csv_LCCDataset(
            args.train_csv_leaf_number_file,
            args.train_csv_leaf_location_file,
            pre_process="keras_like",
            ann_type="count",
            transform=transforms.Compose(
                [
                    Normalizer(pre_process=args.pre_process),
                    Resizer(ann_type="count", min_side=800, max_side=1333),
                ]
            ),
            json_file=args.train_json_file,
        )
    else:
        dataset_train = None

    dataset_val = csv_LCCDataset(
        args.val_csv_leaf_number_file,
        args.val_csv_leaf_location_file,
        pre_process="keras_like",
        ann_type="count",
        transform=transforms.Compose(
            [
                Normalizer(pre_process=args.pre_process),
                Resizer(ann_type="count", min_side=800, max_side=1333),
            ]
        ),
        json_file=args.val_json_file,
        base_dir=args.base_dir,
        have_GT=args.have_GT,
    )
    args.collater = LCC_collater
    return dataset_train, dataset_val


def build_data(args: Any) -> DataBundle:
    """Build datasets, samplers, and dataloaders without changing runner policy."""
    if args.dataset_type == "coco":
        dataset_train, dataset_val = _build_coco_datasets(args)
    elif args.dataset_type == "kcsv" or (
        args.dataset_type == "roots_json"
        and args.network_type
        in ("per_object_attributes", "bbox_detection", "per_object_attributes_multibranch")
    ):
        dataset_train, dataset_val = _build_kcsv_datasets(args)
    elif args.dataset_type == "csv_LCC":
        dataset_train, dataset_val = _build_lcc_datasets(args)
    else:
        raise ValueError(
            "Dataset type not understood (must be csv_LCC or coco), exiting."
        )

    if (
        not getattr(args, "have_GT", True)
        and dataset_val is not None
        and len(dataset_val) == 0
    ):
        raise ValueError(
            f"No input images were found in the no-GT image directory: "
            f"{getattr(args, 'base_dir', None)}"
        )

    sampler = None
    dataloader_train = None
    if args.run_script == "Training":
        sampler = AspectRatioBasedSampler(
            dataset_train,
            batch_size=args.batch_size,
            drop_last=False,
        )
        dataloader_train = DataLoader(
            dataset_train,
            num_workers=args.num_workers,
            collate_fn=args.collater,
            batch_sampler=sampler,
        )

    sampler_val = None
    dataloader_val = None
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val,
            batch_size=1,
            drop_last=False,
            do_shuffle=False,
        )
        dataloader_val = DataLoader(
            dataset_val,
            num_workers=args.num_workers,
            collate_fn=args.collater,
            batch_sampler=sampler_val,
        )

    return DataBundle(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        sampler=sampler,
        sampler_val=sampler_val,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )
