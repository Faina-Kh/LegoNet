"""Weight-loading policy for LegoNet models."""

import os
from typing import Any

import torch

from legonet import config
from legonet.checkpoint_conversion import (
    estimator_module_names,
    validate_estimator_type_against_checkpoint,
)
from legonet.manage_weights import (
    list_checkpoint_modules,
    load_submodule_weights,
    print_module_names,
    validate_checkpoint_modules,
)


def _validate_estimator_weights(
    state_dict: Any,
    args: Any,
    checkpoint_description: str,
) -> None:
    """Reject estimator weights built for a different estimation architecture."""
    if args.network_type in (
        "per_image_estimation",
        "per_object_counting",
    ):
        estimator_modules = estimator_module_names(())
    elif args.network_type in (
        "per_object_attributes",
        "per_object_attributes_multibranch",
    ):
        estimator_modules = estimator_module_names(
            ("length", "diameter", "color")
        )
    else:
        return

    validate_estimator_type_against_checkpoint(
        state_dict,
        estimator_modules,
        args.estimate_type,
        checkpoint_description,
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
        detector_modules = ["backbone_1", "find_1", "where"]
        validate_checkpoint_modules(
            bbox_state,
            detector_modules,
            "Bounding-box checkpoint",
        )

        if args.network_type == "bbox_detection":
            load_submodule_weights(
                model,
                bbox_state,
                submodule_names=detector_modules,
                strict=False,
                verbose=False,
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
                submodule_names=detector_modules,
                strict=False,
                verbose=False,
            )

    if args.load_per_object_counting_weights and args.network_type == "per_object_counting":
        per_object_state = torch.load(
            args.per_object_weights_file,
            map_location=config.General.device,
        )
        print(
            "Available modules in per-object counting weights file:",
            list_checkpoint_modules(per_object_state),
        )
        _validate_estimator_weights(
            per_object_state,
            args,
            "Per-object counting checkpoint",
        )
        if args.estimate_type == "withKeyPoints":
            module_names = ["backbone_2", "find_2", "estimator"]
        elif args.estimate_type == "reg_fpn_p3_p7_min_sig":
            module_names = ["backbone_2", "estimator"]
        else:
            module_names = None

        if module_names is not None:
            validate_checkpoint_modules(
                per_object_state,
                module_names,
                "Per-object counting checkpoint",
            )
            load_submodule_weights(
                model,
                per_object_state,
                submodule_names=module_names,
                strict=False,
                verbose=False,
            )

    if args.load_per_object_attributes_weights and args.network_type in (
        "per_object_attributes",
        "per_object_attributes_multibranch",
    ):
        per_object_state = torch.load(
            args.per_object_weights_file,
            map_location=config.General.device,
        )
        print(
            "Available modules in per-object attributes weights file:\n",
            list_checkpoint_modules(per_object_state),
        )
        _validate_estimator_weights(
            per_object_state,
            args,
            "Per-object attributes checkpoint",
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
            validate_checkpoint_modules(
                per_object_state,
                module_names,
                "Per-object attributes checkpoint",
            )
            load_submodule_weights(
                model,
                per_object_state,
                submodule_names=module_names,
                strict=False,
                verbose=False,
            )


def _load_full_weights(model: Any, args: Any) -> None:
    """Load the configured full-model checkpoint using current module mappings."""
    model_state = torch.load(
        args.full_model_weights,
        map_location=config.General.device,
    )
    checkpoint_modules = list_checkpoint_modules(model_state)
    model_modules = list_checkpoint_modules(model.state_dict())
    print("Available modules in the weights file:\n", checkpoint_modules)
    print("Check keys:")
    print("Checkpoint modules:", checkpoint_modules)
    print("Built model modules:", model_modules)

    validate_checkpoint_modules(
        model_state,
        model_modules,
        "Full-model checkpoint",
    )
    print("Key check passed: checkpoint modules match the built model.\n")

    _validate_estimator_weights(model_state, args, "Full-model checkpoint")

    module_names = None
    if args.network_type == "per_image_estimation":
        module_names = ["backbone", "estimator"]
        if args.estimate_type == "withKeyPoints":
            module_names.insert(1, "find")
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
    elif args.network_type == "per_object_attributes":
        module_names = [
            "bbox_detection",
            "backbone_2",
            "estimator_length",
            "estimator_diameter",
            "estimator_color",
        ]
        if args.estimate_type == "withKeyPoints":
            module_names.insert(2, "find_2")
    elif args.network_type == "per_object_attributes_multibranch":
        module_names = [
            "bbox_detection",
            "backbone_2",
            "find_2",
            "estimator_length",
            "estimator_diameter",
            "estimator_color",
            "find_2_b",
            "backbone_2_b",
        ]
    elif args.network_type == "bbox_detection":
        module_names = ["backbone_1", "find_1", "where"]

    if module_names is not None:
        load_submodule_weights(
            model,
            model_state,
            submodule_names=module_names,
            strict=False,
            verbose=False,
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

