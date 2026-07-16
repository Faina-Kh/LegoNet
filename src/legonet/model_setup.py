"""Weight-loading and legacy-export policy for LegoNet models."""

import os
from typing import Any

import torch

from legonet import config
from legonet.manage_weights import (
    list_checkpoint_modules,
    load_submodule_weights,
    print_module_names,
    save_partial_weights,
)


def _load_partial_weights(model: Any, args: Any) -> None:
    """Load the requested task-specific weights into a model."""
    if args.load_bbox_det_weights:
        bbox_state = torch.load(
            args.bbox_detection_weights_file,
            map_location=config.General.device,
        )
        print(
            "Available modules in bbox_detection weights file:",
            list_checkpoint_modules(bbox_state),
        )

        if args.network_type == "bbox_detection":
            load_submodule_weights(
                model,
                bbox_state,
                submodule_names=["backbone_1", "find_1", "where"],
                strict=False,
                verbose=args.save_from_model_file,
            )
        elif args.network_type in (
            "per_object_counting",
            "per_object_attributes",
            "per_object_attributes_multibranch",
        ):
            print("Available modules in 'bbox_detection' module: ")
            print_module_names(model.bbox_detection)
            load_submodule_weights(
                model.bbox_detection,
                bbox_state,
                submodule_names=["backbone_1", "find_1", "where"],
                strict=False,
                verbose=args.save_from_model_file,
            )

    if args.load_per_object_counting_weights and args.network_type == "per_object_counting":
        per_object_state = torch.load(
            args.per_object_weights_file,
            map_location=config.General.device,
        )
        if args.estimate_type == "withKeyPoints":
            module_names = ["backbone_2", "find_2", "estimator"]
        elif args.estimate_type == "reg_fpn_p3_p7_min_sig":
            module_names = ["backbone_2", "estimator"]
        else:
            module_names = None

        if module_names is not None:
            load_submodule_weights(
                model,
                per_object_state,
                submodule_names=module_names,
                strict=False,
                verbose=args.save_from_model_file,
            )

    if args.load_per_object_attributes_weights and args.network_type in (
        "per_object_attributes",
        "per_object_attributes_multibranch",
    ):
        per_object_state = torch.load(
            args.per_object_weights_file,
            map_location=config.General.device,
        )

        if args.network_type == "per_object_attributes_multibranch":
            module_names = [
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
                "find_2_b",
                "backbone_2_b",
            ]
        elif args.estimate_type == "withKeyPoints":
            module_names = [
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ]
        elif args.estimate_type == "reg_fpn_p3_p7_min_sig":
            module_names = ["backbone_2",
                            "estimator_length",
                            "estimator_diameter",
                            "estimator_color",
                            ]
        else:
            module_names = None

        if module_names is not None:
            load_submodule_weights(
                model,
                per_object_state,
                submodule_names=module_names,
                strict=False,
                verbose=args.save_from_model_file,
            )


def _load_full_weights(model: Any, args: Any) -> None:
    """Load the configured full-model checkpoint using current module mappings."""
    model_state = torch.load(
        args.full_model_weights,
        map_location=config.General.device,
    )
    print(
        "Available modules in the weights file:",
        list_checkpoint_modules(model_state),
    )
    print("Check keys:")

    module_names = None
    if args.network_type == "per_image_estimation_keypoints":
        module_names = ["backbone", "find", "estimator"]
    elif args.network_type == "per_image_estimation_regression":
        module_names = ["backbone", "estimator"]
    elif args.network_type == "per_object_counting":
        if args.estimate_type == "withKeyPoints":
            module_names = [
                "bbox_detection",
                "backbone_2",
                "find_2",
                "estimator",
            ]
        elif args.estimate_type == "reg_fpn_p3_p7_min_sig":
            module_names = ["bbox_detection", "backbone_2", "estimator"]
    elif args.network_type in ("per_object_attributes", "per_object_attributes_multibranch"):
        if args.estimate_type == "withKeyPoints":
            module_names = ["bbox_detection", "per_object_attributes"]
    elif args.network_type == "bbox_detection":
        module_names = ["backbone_1", "find_1", "where"]

    if module_names is not None:
        load_submodule_weights(
            model,
            model_state,
            submodule_names=module_names,
            strict=False,
            verbose=args.save_from_model_file,
        )


def load_requested_weights(model: Any, args: Any) -> None:
    """Load partial or full weights according to the current run arguments."""
    print("Loading weights from: ", os.path.join(args.myExpPath, "Weights \n"))
    print("All available modules in legoNet: ")
    print_module_names(model)

    if args.load_partial_weights:
        _load_partial_weights(model, args)
    elif args.load_full_model_weights:
        _load_full_weights(model, args)


def export_legacy_weights(model: Any, args: Any) -> None:
    """Convert a legacy model checkpoint into task-specific weight files."""
    if args.network_type == "per_object_attributes_multibranch":
        file_model = {
            "file_bbox": torch.load(
                args.model_path["bbox_path"],
                map_location=config.General.device,
            ),
            "file_limit5Path": torch.load(
                args.model_path["limit5Path"],
                map_location=config.General.device,
            ),
            "file_all_3setsPath": torch.load(
                args.model_path["all_3setsPath"],
                map_location=config.General.device,
            ),
        }
    else:
        file_model = torch.load(
            args.model_path,
            map_location=config.General.device,
        )
        print(
            "Available modules in model file:",
            list_checkpoint_modules(file_model.state_dict()),
        )

    if args.network_type == "bbox_detection":
        save_partial_weights(
            args,
            model,
            file_model,
            tasks=["bbox_detection"],
        )
    elif args.network_type == "per_object_counting":
        save_partial_weights(
            args,
            model,
            file_model,
            tasks=["bbox_detection", "per_object_counting"],
            output_name=args.output_name,
        )
    elif args.network_type in ("per_object_attributes", "per_object_attributes_multibranch"):
        save_partial_weights(
            args,
            model,
            file_model,
            tasks=["bbox_detection", "per_object_attributes"],
            output_name=args.output_name,
        )
    elif args.network_type in ("per_image_estimation_keypoints", "per_image_estimation_regression"):
        save_partial_weights(
            args,
            model,
            file_model,
            tasks=["per_image_attributes"],
            output_name=args.output_name,
        )
