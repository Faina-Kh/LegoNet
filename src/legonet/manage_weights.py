"""Utilities for inspecting, extracting, and loading LegoNet checkpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

def rename_state_dict_keys(
    state_dict: Mapping[str, Any],
    rename_map: Mapping[str, str],
) -> Mapping[str, Any]:
    """Rename matching dot-separated components in state-dict keys."""
    if len(state_dict)>0:
        new_state_dict = {}
        for k, v in state_dict.items():
            parts = k.split(".")
            parts = [rename_map.get(p, p) for p in parts]
            new_k = ".".join(parts)
            new_state_dict[new_k] = v
    else:
        new_state_dict = state_dict

    return new_state_dict

def list_checkpoint_modules(state_dict: Mapping[str, Any]) -> list[str]:
    """Return the sorted top-level module names stored in a state dict."""
    return sorted(set(k.split('.')[0] for k in state_dict.keys()))


def validate_checkpoint_modules(
    state_dict: Mapping[str, Any],
    expected_modules: Sequence[str],
    checkpoint_description: str,
) -> None:
    """Raise when checkpoint modules do not match the requested architecture."""
    actual = set(list_checkpoint_modules(state_dict))
    expected = set(expected_modules)
    if actual == expected:
        return

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing modules: {missing}")
    if unexpected:
        details.append(f"unexpected modules: {unexpected}")
    raise ValueError(
        f"{checkpoint_description} architecture does not match the built model "
        f"({'; '.join(details)} \n). "
        f"Select weights created for the same network type and estimate type."
    )

def save_partial_weights(
    args: Any,
    model: Any,
    file_model: Any,
    tasks: Sequence[str] = (),
    output_name: str = "",
) -> None:
    """Extract and save task-specific weights using the legacy module mapping."""
    import torch

    for task in tasks:
        if task == "bbox_detection":
            submodules = ['backbone_1', 'find_1', 'where']
            rename_map = {}
            save_path = os.path.join(args.bbox_detection_weights_dir, 'legonet_' + 'bbox_'+args.dataset_name + '.pt')

        if task == "per_object_counting":
            if args.estimate_type =='withKeyPoints':
                submodules = ['backbone_2', 'find_2', 'LeanCountingModule']

                rename_map = {'LeanCountingModule': 'estimator'}

            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                submodules = ['backbone_2', 'find_2', 'CountWithRegModule']

                rename_map = {'CountWithRegModule': 'estimator'}

            save_path = os.path.join(args.per_object_weights_dir, 'legonet_' + output_name + '.pt')
            # torch.save(filtered_state_dict, os.path.join(args.per_object_weights_dir,
            #                                              'legonet_per_object_count_epoch=249.pt'))

        if task == "per_image_attributes":
            if args.estimate_type == "withKeyPoints":
                submodules = ['backbone_1', 'find_2', 'LeanCountingModule']

                rename_map = {'backbone_1': 'backbone',
                              'find_2': 'find',
                              'LeanCountingModule': 'estimator'}

            else:
                submodules = ['backbone_1', 'CountWithRegModule', 'CountRegSubmodel']

                rename_map = {'backbone_1': 'backbone',
                              'CountWithRegModule': 'estimator',
                              'CountRegSubmodel': 'regSubmodel'}

            save_path = os.path.join(args.per_image_weights_dir, 'legonet_'+output_name+'.pt')

        if task == "per_object_attributes":
            if args.estimate_type =='withKeyPoints':

                rename_map = {'LeanCountingModule_length': 'estimator_length',
                              'LeanCountingModule_diameter': 'estimator_diameter',
                              'LeanCountingModule_color': 'estimator_color'}

                if args.network_type == "per_object_attributes_multibranch":

                    submodules_limit5Path = {"backbone_2", "find_2",
                                             "LeanCountingModule_color", "LeanCountingModule_diameter"}

                    submodules_all_3setsPath = {"backbone_2", "find_2", "LeanCountingModule_length"}

                else:
                    submodules = ['backbone_2', 'find_2',
                                  'LeanCountingModule_length', 'LeanCountingModule_diameter',
                                  'LeanCountingModule_color']

            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                submodules = ['backbone_2',
                              'CountWithRegModule_length', 'CountWithRegModule_diameter', 'CountWithRegModule_color']

                rename_map = {'CountWithRegModule_length': 'estimator_length',
                              'CountWithRegModule_diameter': 'estimator_diameter',
                              'CountWithRegModule_color': 'estimator_color',
                              'CountRegSubmodel': 'regSubmodel'}

            save_path = os.path.join(args.per_object_weights_dir, 'legonet_' + output_name + '.pt')

            # torch.save(filtered_state_dict, os.path.join(args.per_object_weights_dir,
            #                                              'legonet_per_object_attr_epoch=90.pt'))

        if args.network_type == "per_object_attributes_multibranch":
            if task == "bbox_detection":
                file_bbox = file_model["file_bbox"]
                print("Available modules in bbox file:", list_checkpoint_modules(file_bbox.state_dict()))

                filtered_state_dict = {
                    k: v for k, v in file_bbox.state_dict().items()
                    if any(k == m or k.startswith(m + ".") for m in submodules)
                }

                renamed_state_dict = rename_state_dict_keys(filtered_state_dict, rename_map)
                print("Available modules in renamed_state_dict:", list_checkpoint_modules(renamed_state_dict))

            else: #"per_object_attributes"

                file_limit5Path = file_model["file_limit5Path"]
                file_all_3setsPath = file_model["file_all_3setsPath"]

                print("Available modules in limit5Path file:", list_checkpoint_modules(file_limit5Path.state_dict()))
                print("Available modules in all_3setsPath file:", list_checkpoint_modules(file_all_3setsPath.state_dict()))

                filtered_state_dict = {
                    k: v for k, v in file_limit5Path.state_dict().items()
                    if any(k == m or k.startswith(m + ".") for m in submodules_limit5Path)
                }

                renamed_state_dict = rename_state_dict_keys(filtered_state_dict, rename_map)
                print("Available modules in renamed_state_dict:", list_checkpoint_modules(renamed_state_dict))

                filtered_state_dict_2 = {
                    k: v for k, v in file_all_3setsPath.state_dict().items()
                    if any(k == m or k.startswith(m + ".") for m in submodules_all_3setsPath)
                }

                rename_map_2 = {"backbone_2": "backbone_2_b",
                              "find_2": "find_2_b",
                              "LeanCountingModule_length": "estimator_length"}

                renamed_state_dict_2 = rename_state_dict_keys(filtered_state_dict_2, rename_map_2)
                print("Available modules in renamed_state_dict_2:", list_checkpoint_modules(renamed_state_dict_2))

                renamed_state_dict.update(renamed_state_dict_2)
                print("Available modules in renamed_state_dict:", list_checkpoint_modules(renamed_state_dict))

        else:
            filtered_state_dict = {
                k: v for k, v in file_model.state_dict().items()
                if any(k == m or k.startswith(m + ".") for m in submodules)
            }
            # if args.dataset_name == 'grapes' and task == "bbox_detection":
            #     filtered_state_dict = {
            #         k: v for k, v in file_model.state_dict().items()
            #         if any(k == m or k.startswith(m + ".") for m in submodules)
            #     }
            #     filtered_state_dict = {  # bbox_detection_state_dict
            #         k.replace("bbox_detection.", "", 1): v
            #         for k, v in file_model.items()
            #         if k.startswith("bbox_detection.")
            #     }
            # else:
            #     filtered_state_dict = {
            #         k: v for k, v in file_model.state_dict().items()
            #         if any(k == m or k.startswith(m + ".") for m in submodules)
            #     }

            if args.network_type != "bbox_detection":
                renamed_state_dict = rename_state_dict_keys(filtered_state_dict, rename_map)
                print("Available modules in renamed_state_dict:", list_checkpoint_modules(renamed_state_dict))
            else:
                renamed_state_dict = filtered_state_dict

        if args.network_type != "bbox_detection" and task == "bbox_detection":
            load_result = model.bbox_detection.load_state_dict(renamed_state_dict, strict=False)
        else:
            load_result = model.load_state_dict(renamed_state_dict, strict=False)

        clean_state_dict = {
            k: v for k, v in renamed_state_dict.items()
            if k not in load_result.unexpected_keys
        }
        print("Available modules in clean_state_dict:", list_checkpoint_modules(clean_state_dict))

        torch.save(clean_state_dict, save_path)


def load_submodule_weights(
    model: Any,
    state_dict: Mapping[str, Any],
    submodule_names: Sequence[str],
    strict: bool = True,
    verbose: bool = False,
) -> None:
    """Load matching checkpoint entries into selected model submodules."""
    """
    Load weights of a specific submodule from a full state_dict.

    Args:
        model: full model
        state_dict: loaded state_dict (from torch.load)
        submodule_name: str, e.g. 'backbone'
        strict: passed to load_state_dict
        verbose: print debug info

    Returns:
        missing_keys, unexpected_keys
    """
    for submodule_name in submodule_names:
        # 1. get submodules
        submodule = dict(model.named_modules()).get(submodule_name, None)

        if submodule is None:
            raise ValueError(f"Submodule '{submodule_name}' not found in model")

        # 2. filter relevant weights
        prefix = submodule_name + "."
        filtered_dict = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

        if verbose:
            print(f"\nLoading '{submodule_name}'")
            print(f"Found {len(filtered_dict)} matching parameters")

        if len(filtered_dict) == 0:
            raise ValueError(f"No weights found for submodule '{submodule_name}'")

        # 3. load per task
        result = submodule.load_state_dict(filtered_dict, strict=strict)

        if verbose:
            print(f"Missing keys: {result.missing_keys}")
            print(f"Unexpected keys: {result.unexpected_keys} \n")

    #return result


def print_module_names(model: Any) -> None:
    """Print immediate child modules and mark those without checkpoint state."""
    names = []
    for name, module in model.named_children():
        has_weights = bool(module.state_dict())
        names.append(name if has_weights else f"{name} (no weights)")
    print(names, '\n')

if __name__ == '__main__':


    ##################################################

    weights_path = '' #'C:/Users/bordezki/Desktop/LegoNet/ExpResults/grapes_detection_filterBBOX_minScore_0.7_prev/Weights'



    #################################################
    """
    # change
    torch.save(legonet.state_dict(), weights_path)
    
    to 
    torch.save({"model": legonet.state_dict()}, "model.pth")
    """





