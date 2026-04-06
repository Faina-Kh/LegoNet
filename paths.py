import os
from pathlib import Path


def get_paths(STORAGE_PATH, dataset_name):

    paths = {}

    if dataset_name == "grapes":
        DATASETS_PATH = os.path.join(STORAGE_PATH, "Datasets","grapes")
    elif dataset_name == "roots":
        DATASETS_PATH = os.path.join(STORAGE_PATH, "Datasets","roots")

    MODELS_PATH = os.path.join(STORAGE_PATH, "Models")
    EXP_RESULTS_PATH = os.path.join(STORAGE_PATH, "ExpResults")

    paths["DATASETS_PATH"] = DATASETS_PATH
    #paths["MODELS_PATH"] = MODELS_PATH
    paths["EXP_RESULTS_PATH"] = EXP_RESULTS_PATH

    for d in paths.keys():
        Path(paths[d]).mkdir(exist_ok=True)

    return paths



