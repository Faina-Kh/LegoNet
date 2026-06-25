import os
import torch





def rename_state_dict_keys(state_dict, rename_map):
    """
    rename_map: dict like {"old_name": "new_name"}
    """
    if len(state_dict)>0:
        new_state_dict = {}

        # for k, v in state_dict.items():
        #     new_k = k
        #     for old, new in rename_map.items():
        #         if k.startswith(old):
        #             new_k = k.replace(old, new, 1)
        #             break
        #     new_state_dict[new_k] = v

        for k, v in state_dict.items():
            parts = k.split(".")
            parts = [rename_map.get(p, p) for p in parts]
            new_k = ".".join(parts)
            new_state_dict[new_k] = v
    else:
        new_state_dict = state_dict

    return new_state_dict

def list_checkpoint_modules(state_dict):
    return sorted(set(k.split('.')[0] for k in state_dict.keys()))

def save_partial_weights(args, model, file_model, tasks = [], output_name=""):
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
            if args.network_type == "counting_lean":
                submodules = ['backbone_1', 'find_2', 'LeanCountingModule']

                rename_map = {'backbone_1': 'backbone',
                              'find_2': 'find',
                              'LeanCountingModule': 'estimator'}

            elif args.network_type == "counting_reg":
                submodules = ['backbone_1', 'CountWithRegModule', 'CountRegSubmodel']

                rename_map = {'backbone_1': 'backbone',
                              'CountWithRegModule': 'estimator',
                              'CountRegSubmodel': 'regSubmodel'}

            save_path = os.path.join(args.per_image_weights_dir, 'legonet_'+output_name+'.pt')

        if task == "per_object_attributes":# and args.network_type == "both_for_roots_2":
            if args.estimate_type =='withKeyPoints':

                rename_map = {'LeanCountingModule_length': 'estimator_length',
                              'LeanCountingModule_diameter': 'estimator_diameter',
                              'LeanCountingModule_color': 'estimator_color'}

                if args.network_type == "both_Back2bFind2b":

                    submodules_limit5Path = {"backbone_2", "find_2",
                                             "LeanCountingModule_color", "LeanCountingModule_diameter"}

                    submodules_all_3setsPath = {"backbone_2", "find_2", "LeanCountingModule_length"}

                else:
                    submodules = ['backbone_2', 'find_2',
                                  'LeanCountingModule_length', 'LeanCountingModule_diameter',
                                  'LeanCountingModule_color']

            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                submodules = ['backbone_2', 'find_2',
                              'CountWithRegModule_length', 'CountWithRegModule_diameter', 'CountWithRegModule_color']

                rename_map = {'CountWithRegModule_length': 'estimator_length',
                              'CountWithRegModule_diameter': 'estimator_diameter',
                              'CountWithRegModule_color': 'estimator_color',
                              'CountRegSubmodel': 'regSubmodel'}

            save_path = os.path.join(args.per_object_weights_dir, 'legonet_' + output_name + '.pt')

            # torch.save(filtered_state_dict, os.path.join(args.per_object_weights_dir,
            #                                              'legonet_per_object_attr_epoch=90.pt'))

        if args.network_type == "both_Back2bFind2b":
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


def load_submodule_weights(model, state_dict, submodule_names, strict=True, verbose=False):
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


def print_module_names(model):
    names = []
    for name, module in model.named_children():
        names.append(name)
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





