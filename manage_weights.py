import os
import torch



def rename_state_dict_keys(state_dict, rename_map):
    """
    rename_map: dict like {"old_name": "new_name"}
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k
        for old, new in rename_map.items():
            if k.startswith(old):
                new_k = k.replace(old, new, 1)
                break
        new_state_dict[new_k] = v

    return new_state_dict

def list_checkpoint_modules(state_dict):
    return sorted(set(k.split('.')[0] for k in state_dict.keys()))


def load_submodule_weights(model, state_dict, submodule_name, strict=True, verbose=True):
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

    # 1. get submodule
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

    # 3. load
    result = submodule.load_state_dict(filtered_dict, strict=strict)

    if verbose:
        print("Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)

    return result




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0")

    ##################################################

    weights_path = 'C:/Users/bordezki/Desktop/LegoNet/ExpResults/grapes_detection_filterBBOX_minScore_0.7_prev/Weights'

    model_path = ""

    #################################################
    """
    # change
    torch.save(legonet.state_dict(), weights_path)
    
    to 
    torch.save({"model": legonet.state_dict()}, "model.pth")
    """



