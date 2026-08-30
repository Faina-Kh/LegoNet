from __future__ import annotations

import os
import sys
import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from legonet import config, paths
from legonet.streamlit_output import append_evaluation_summary
from legonet.pretrained import resolve_pretrained_weights
from legonet.datasets import default_storage_root, ensure_dataset_available
from legonet.checkpoint_conversion import (
    ESTIMATE_TYPE_CHOICES,
    normalize_estimate_type,
)
from datetime import datetime

import warnings
import faulthandler

########################################################################################################################

def finalize_text_results(args, execution_time: float) -> None:
    """Append execution time and the consolidated summary to the text output."""
    if getattr(args, "txt_results", ""):
        append_evaluation_summary(args.txt_results)

        with open(args.txt_results, "a", encoding="utf-8") as results_file:
            results_file.write(
                f"\n"
                f"Execution time in minutes: {(execution_time / 60):.2f}\n"
            )


def parse_bool(value):
    """Parse a command-line boolean value."""
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False

    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without applying runtime defaults."""
    parser = argparse.ArgumentParser(description='main script.')

    parser.add_argument("--gpu-num", "--gpu_num", default=None, help="CUDA device index exposed to the run.")
    parser.add_argument(
        "--storage-path",
        "--STORAGE_PATH",
        "--storage_path",
        default=None,
        help="Root storage path containing Datasets and ExpResults.",
    )
    parser.add_argument("--dataset-name", "--dataset_name", choices=["grapes", "roots"], default=None)
    parser.add_argument(
        "--network-type",
        "--network_type",
        choices=[
            "bbox_detection",
            "per_image_estimation",
            "per_object_counting",
            "per_object_attributes",
            "per_object_attributes_multibranch",
        ],
        default=None,
    )
    parser.add_argument("--current-results-dir", "--current_results_dir", default=None)
    parser.add_argument(
        "--estimate-type",
        "--estimate_type",
        choices=ESTIMATE_TYPE_CHOICES,
        default=None,
        help=(
            "Estimator architecture: keypoints or regression."
        ),
    )
    parser.add_argument("--run-script", "--run_script", choices=["Training", "Inference"], default=None)
    parser.add_argument("--val-set", "--val_set", choices=["Val", "Test"], default=None)
    parser.add_argument("--num-of-epochs", "--num_of_epochs", type=int, default=None)
    parser.add_argument("--have-gt", "--have_GT", type=parse_bool, default=None)
    parser.add_argument("--to-draw", "--to_draw", type=parse_bool, default=None)
    parser.add_argument(
        "--draw-detection-overview",
        "--draw_detection_overview",
        type=parse_bool,
        default=None,
    )
    parser.add_argument(
        "--draw-gt-only",
        "--draw_gt_only",
        type=parse_bool,
        default=None,
    )
    parser.add_argument(
        "--draw-individual-object-visualizations",
        "--draw_individual_object_visualizations",
        type=parse_bool,
        default=None,
        help=(
            "Save separate per-object GT and predicted-box images. Defaults "
            "to true for roots and false for grapes. Predicted crops and "
            "keypoint heatmaps are unaffected."
        ),
    )
    parser.add_argument("--save-from-model-file", "--save_from_model_file", type=parse_bool, default=None)
    parser.add_argument("--load-weights", "--load_weights", type=parse_bool, default=None)
    parser.add_argument(
        "--load-only-bbox-weights",
        "--load_only_bbox_weights",
        type=parse_bool,
        default=None,
        help="Load the pretrained detector while initializing per-object heads from scratch.",
    )
    parser.add_argument("--evaluate-detection", "--evaluate_detection", type=parse_bool, default=None)
    parser.add_argument(
        "--compare-keypoint-protocols",
        "--compare_keypoint_protocols",
        type=parse_bool,
        default=None,
        help=(
            "Write diagnostic keypoint AP results for the current protocol, "
            "processed-map local maxima, and the historical raw-map protocol."
        ),
    )
    parser.add_argument("--weights-type", "--weights_type", choices=['full_model_weights', 'partial_weights'], default=None)
    parser.add_argument(
        "--weights-mode",
        "--weights_mode",
        choices=["auto", "none", "full", "partial", "detector_only"],
        default=None,
        help=("Select checkpoint loading. 'auto' downloads the matching published "
              "checkpoint when it is not already cached."),
    )
    parser.add_argument("--full-weights-file", "--full_weights_file", default=None)
    parser.add_argument("--bbox-weights-file", "--bbox_weights_file", default=None)
    parser.add_argument("--per-object-weights-file", "--per_object_weights_file", default=None)
    parser.add_argument(
        "--download-missing-data",
        "--download_missing_data",
        type=parse_bool,
        default=None,
        help="Automatically download and prepare a missing public dataset (default: true).",
    )
    parser.add_argument(
        "--checkpoint-attribute",
        "--checkpoint_attribute",
        choices=["length", "diameter", "color"],
        default=None,
        help=(
            "Attribute whose error selects the best attribute-model checkpoint. "
            "Defaults to length. Counting always uses relative error."
        ),
    )

    parsed_args = parser.parse_args(args)
    # if parsed_args.run_script is None:
    #     parsed_args.run_script = parsed_args.run_script_positional

    return parsed_args


def get_weights_file(weights_dir):
    files = [p for p in Path(weights_dir).iterdir() if p.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one weights file in {weights_dir}, found {len(files)}."
        )
    return str(files[0])


def require_weights_file(value: str | None, option_name: str) -> str:
    """Resolve one explicitly supplied checkpoint path or fail clearly."""
    if not value:
        raise ValueError(f"{option_name} is required for the selected weights mode.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Weights file does not exist: {path}")
    return str(path.resolve())


def configure_weights_mode(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize legacy loading flags into one explicit weights mode."""
    mode = args.weights_mode
    if mode is None:
        if args.run_script == "Inference":
            mode = "auto"
        elif args.load_only_bbox_weights is True:
            mode = "detector_only"
        elif (
            args.load_only_bbox_weights is None
            and args.run_script == "Training"
            and args.network_type in PER_OBJECT_NETWORKS
        ):
            mode = "detector_only"
        elif not args.load_weights:
            mode = "none"
        elif args.weights_type == "full_model_weights":
            mode = "full"
        else:
            mode = "partial"

    args.weights_mode = mode
    args.load_only_bbox_weights = mode == "detector_only"
    args.load_weights = mode in ("full", "partial")
    args.weights_type = (
        "full_model_weights" if mode == "full" else "partial_weights"
    )
    return args


def resolve_storage_path(
    cli_value: str | None,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve and validate the storage root from CLI or environment input."""
    environment = os.environ if environment is None else environment
    raw_path = cli_value or environment.get("LEGONET_STORAGE_PATH")

    if not raw_path:
        default_root = default_storage_root()
        if default_root is None:
            raise ValueError(
                "LegoNet storage path is required outside a source checkout. "
                "Pass --storage-path PATH or set LEGONET_STORAGE_PATH."
            )
        return str(default_root)

    storage_path = Path(raw_path).expanduser()
    if storage_path.exists() and not storage_path.is_dir():
        raise ValueError(
            "LegoNet storage path is not a directory: "
            f"{storage_path}"
        )
    storage_path.mkdir(parents=True, exist_ok=True)

    return str(storage_path)


def resolve_boolean_options(args: argparse.Namespace) -> argparse.Namespace:
    """Apply boolean defaults while preserving explicit CLI values."""
    args.have_GT = True if args.have_gt is None else args.have_gt
    args.to_draw = False if args.to_draw is None else args.to_draw
    args.draw_detection_overview = (
        True
        if args.draw_detection_overview is None
        else args.draw_detection_overview
    )
    args.draw_gt_only = (
        False if args.draw_gt_only is None else args.draw_gt_only
    )
    args.draw_individual_object_visualizations = (
        args.dataset_name == "roots"
        if args.draw_individual_object_visualizations is None
        else args.draw_individual_object_visualizations
    )
    args.evaluate_detection = (
        True if args.evaluate_detection is None else args.evaluate_detection
    )
    args.compare_keypoint_protocols = (
        False
        if args.compare_keypoint_protocols is None
        else args.compare_keypoint_protocols
    )
    args.load_weights = True if args.load_weights is None else args.load_weights
    args.save_from_model_file = (
        False
        if args.save_from_model_file is None
        else args.save_from_model_file
    )
    args.download_missing_data = (
        True
        if args.download_missing_data is None
        else args.download_missing_data
    )

    if args.load_weights and args.save_from_model_file:
        raise ValueError(
            "--load-weights and --save-from-model-file cannot both be true."
        )

    return args


DEFAULT_ESTIMATE_TYPE_BY_NETWORK = {
    "per_image_estimation": "withKeyPoints",
    "per_object_attributes_multibranch": "withKeyPoints",
    #"per_object_counting": "withKeyPoints" # dont have currently weights for "reg_fpn_p3_p7_min_sig"
}

MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS = ("bbox_detection")

PER_OBJECT_NETWORKS = ("per_object_counting", "per_object_attributes", "per_object_attributes_multibranch")

INCLUDE_BBOX_DETECTION = ("bbox_detection", "per_object_counting", "per_object_attributes", "per_object_attributes_multibranch")


NETWORKS_OPTIONS_BY_DATASETS = {'roots': ("bbox_detection", "per_image_estimation", "per_object_attributes",
                                          "per_object_attributes_multibranch"),
                                'grapes': ("bbox_detection", "per_object_counting")
                                }

SUPPORTED_ESTIMATE_TYPES_BY_NETWORK = {
    "bbox_detection": ("withKeyPoints", "reg_fpn_p3_p7_min_sig"),
    "per_image_estimation": ("withKeyPoints", "reg_fpn_p3_p7_min_sig"),
    "per_object_counting": ("withKeyPoints", "reg_fpn_p3_p7_min_sig"),
    "per_object_attributes": ("withKeyPoints", "reg_fpn_p3_p7_min_sig"),
    "per_object_attributes_multibranch": ("withKeyPoints",),
}


def validate_configuration(args: argparse.Namespace) -> argparse.Namespace:
    """Validate supported public experiment-option combinations."""
    args.estimate_type = normalize_estimate_type(args.estimate_type)
    supported_networks = NETWORKS_OPTIONS_BY_DATASETS.get(args.dataset_name)
    if supported_networks is None:
        raise ValueError(f"Unsupported dataset: {args.dataset_name!r}.")

    if args.network_type not in supported_networks:
        choices = ", ".join(supported_networks)
        raise ValueError(
            f"Network type {args.network_type!r} is not supported for dataset "
            f"{args.dataset_name!r}. Choose one of: {choices}."
        )

    supported_estimates = SUPPORTED_ESTIMATE_TYPES_BY_NETWORK[args.network_type]
    if args.estimate_type not in supported_estimates:
        choices = ", ".join(supported_estimates)
        raise ValueError(
            f"Estimate type {args.estimate_type!r} is not supported for network "
            f"{args.network_type!r}. Choose one of: {choices}."
        )

    if args.run_script not in ("Training", "Inference"):
        raise ValueError(f"Unsupported run mode: {args.run_script!r}.")
    if args.val_set not in ("Val", "Test"):
        raise ValueError(f"Unsupported validation set: {args.val_set!r}.")

    if args.run_script == "Training":
        if not args.have_GT:
            raise ValueError("Training requires --have-gt true.")
        if args.val_set != "Val":
            raise ValueError("Training requires --val-set Val.")

    if getattr(args, "load_only_bbox_weights", False) and not (
        args.run_script == "Training" and args.network_type in PER_OBJECT_NETWORKS
    ):
        raise ValueError(
            "--load-only-bbox-weights is available only when training a "
            "per-object network."
        )
    if (
        getattr(args, "load_only_bbox_weights", False)
        and getattr(args, "save_from_model_file", False)
    ):
        raise ValueError(
            "--load-only-bbox-weights and --save-from-model-file cannot both be true."
        )
    if (
        getattr(args, "weights_mode", None) == "partial"
        and args.network_type
        == "per_image_estimation"
    ):
        raise ValueError(
            "Per-image estimation networks require --weights-mode full or none."
        )

    return args


def initialize_dataset_runtime_flags(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Set dataset-wide inference flags before model-specific configuration."""
    # Roots models support inference on images without annotated objects.
    # Grapes per-object counting does not, so this remains false for that task.
    args.predict_empty_image = args.dataset_name == "roots"
    args.do_nmcs = args.dataset_name == "grapes"
    return args


def configure_runtime(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the legacy runtime configuration without starting a run."""
    time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    ########################################################################################################################
    # user definitions
    ########################################################################################################################
    args.gpu_num = args.gpu_num or '0'

    args.STORAGE_PATH = resolve_storage_path(args.storage_path)
    args.dataset_name = args.dataset_name or "roots" #"grapes" #"roots"
    args.network_type = args.network_type or "per_object_attributes_multibranch"
    selected_estimate_type = args.estimate_type or DEFAULT_ESTIMATE_TYPE_BY_NETWORK.get(
        args.network_type,
        "withKeyPoints",
    )
    args.estimate_type = normalize_estimate_type(selected_estimate_type)
    resolve_boolean_options(args)

    args.run_script = args.run_script or 'Inference' #'Training' #'Inference'
    args.val_set = args.val_set or "Test" #"Test" #"Val"

    args.num_of_epochs = args.num_of_epochs or 300
    configure_weights_mode(args)
    # Reject unsupported experiment combinations before any network access.
    validate_configuration(args)
    resolve_pretrained_weights(args)
    # Automatic resolution converts ``auto`` into a concrete loading mode.
    configure_weights_mode(args)
    validate_configuration(args)
    type_name = '_KP_' if args.estimate_type == "withKeyPoints" else "_Reg_"

    #################################################
    args.choose_epoch_by_IoUavg = False
    #config.Detection.min_score = 0.05 # not 0.7
    #################################################

    if args.run_script == 'Training':
        args.current_results_dir = args.current_results_dir or (args.network_type + type_name + 'Training')
    else:
        args.current_results_dir = args.current_results_dir or (args.network_type +"_"+ type_name + args.val_set ) #+ "_Check" ) #+ type_name) # + type_name # +'_by_IoUavg_'+ 'new_84' ) #'_'+time_stemp

    args.evaluate_per_object = args.network_type in PER_OBJECT_NETWORKS

    if (
        args.run_script == "Training"
        and args.network_type in PER_OBJECT_NETWORKS
        and not (args.load_weights or args.load_only_bbox_weights)
    ):
        raise ValueError(
            "Per-object training requires pretrained detector weights. "
            "Use --load-only-bbox-weights true to train a new estimation head."
        )

    args.load_full_model_weights = args.weights_mode == "full"
    args.load_partial_weights = args.weights_mode in ("partial", "detector_only")

    args.load_bbox_det_weights = (
        args.weights_mode in ("partial", "detector_only")
        and args.network_type in INCLUDE_BBOX_DETECTION
    )

    args.load_per_object_counting_weights = (
        args.weights_mode == "partial"
        and args.network_type == "per_object_counting"
    )
    args.load_per_object_attributes_weights = (
        args.weights_mode == "partial"
        and args.network_type
        in ("per_object_attributes", "per_object_attributes_multibranch")
    )



    ########################################################################################################################
    # GPU settings
    ########################################################################################################################

    import numpy as np
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f'Running on physical GPU {args.gpu_num}\n')
    else:
        device = torch.device("cpu")

    config.General.device = device

    ######################################################################
    # General paths
    ######################################################################

    myPaths = paths.get_paths(args.STORAGE_PATH, args.dataset_name)
    myDatasetsPath = myPaths["DATASETS_PATH"]
    args.myExpPath = myPaths["EXP_RESULTS_PATH"]
    ensure_dataset_available(
        args.dataset_name,
        myDatasetsPath,
        download_missing=args.download_missing_data,
    )

    config.General.experiment_path = os.path.join(args.myExpPath, 'Results', args.current_results_dir)
    os.makedirs(config.General.experiment_path, exist_ok=True)

    config.General.dataset_name= args.dataset_name
    config.General.MODE = args.run_script
    config.General.model_name = args.network_type
    config.Detect_and_Estimate.type = args.network_type
    config.AttributeEstimation.calc_det_performance = args.evaluate_detection

    # task specific definitions - ToDo - organize per task type
    args.freeze_detection = True
    args.eval_in_train = True

    args.eval_detection_params = False

    if args.network_type != "bbox_detection":
        args.freeze_detection = True

    initialize_dataset_runtime_flags(args)

    if args.dataset_name == 'grapes':
        # ``filter_empty_bbox=True`` removes images without berry points and
        # boxes without associated points from the dataset. In contrast,
        # ``predict_empty_image=False`` prevents per-object counting if an
        # empty annotated sample nevertheless reaches the model.
        args.filter_empty_bbox = True
        config.General.filter_empty_bbox = args.filter_empty_bbox

        args.predict_empty_image = False

        args.do_nmcs = True


    elif args.dataset_name == 'roots':

        args.filter_empty_bbox = False
        config.General.filter_empty_bbox = args.filter_empty_bbox

        if args.network_type in INCLUDE_BBOX_DETECTION:
            config.Detection.change_anchors = True
            config.Detection.ratios = np.array([0.5, 1, 3])  # np.array([0.5, 1, 3]) #np.array([0.5, 1, 4]) #np.array([0.5, 1, 2])
            print("Detection.ratios:", config.Detection.ratios)

            args.predict_empty_image = True

            args.do_nmcs = False

            args.dia_loss_weight = 10  # 10 #1000 #10 #100
            args.color_loss_weight = 100  # 100
            args.maps_loss_weight = 1 #10

            # #os.path.join(args.dataset_path,"Results\\detection\\IOU 0.7_score 0.7_limit_5\\2023-02-15_154227","saved_weights_epoch_280")
            # "IOU 0.9_score 0.7_limit_5\\2023-02-16_150507" ,"saved_weights_epoch_158")
            # "Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", " saved_weights_epoch_175")
            # "C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults\\KK_Exp_Results\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
            # #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')

    config.General.predict_empty_image = args.predict_empty_image
    config.AttributeEstimation.do_nmcs = args.do_nmcs
    config.AttributeEstimation.estimate_type = args.estimate_type
    ######################################################################
    # ToDo - to resolve
    ######################################################################

    args.num_workers = 0 # 0 - for single processing
    args.batch_size = 1
    args.output_size = 1

    args.pre_process = 'torch_like' #'keras_like'  # torch_like
    args.backbone_type = "ResNetBackboneModule"

    args.loss_weight = 1  # 1  #1000 #10 #100 # roots_both ablations

    if args.run_script == 'Training':
        args.epochs = args.num_of_epochs
        config.General.to_draw = False
        config.DrawProperties.DRAW_MAPS = False
        config.AttributeEstimation.calc_det_performance = False

    elif args.to_draw:
        config.General.to_draw = True
        if args.estimate_type == 'withKeyPoints':
            config.DrawProperties.DRAW_MAPS = True
            config.AttributeEstimation.calc_det_performance = True
        else:
            config.DrawProperties.DRAW_MAPS = False
            config.AttributeEstimation.calc_det_performance = False


    ########################################################################################################################
    # Define the data format and network options
    ########################################################################################################################

    if args.dataset_name == "grapes":
        args.dataset_type = "kcsv"

    elif args.dataset_name == "roots":
        if args.network_type == "per_image_estimation":
            args.dataset_type = 'csv_LCC'
        else:
            args.dataset_type = "roots_json"

    if args.network_type == "bbox_detection":
        config.General.NETWORK_TYPE = config.NetworkType.detection
    elif args.network_type == "per_image_estimation":
        config.General.NETWORK_TYPE = config.NetworkType.per_image_estimation
    elif args.network_type == "per_object_counting" or args.network_type == "per_object_attributes" or args.network_type == "per_object_attributes_multibranch":
        config.General.NETWORK_TYPE = config.NetworkType.detection_and_estimation

    ########################################################################################################################
    # Define the paths
    ########################################################################################################################

    if args.run_script == 'Training':
        config.General.weights_dir = os.path.join(config.General.experiment_path, "Weights")
        #time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        config.General.weights_dir += "\\"+time_stemp
        os.makedirs(config.General.weights_dir, exist_ok=True)

    if args.dataset_type == 'kcsv':
        args.kcsv_train = os.path.join(myDatasetsPath, 'train.txt') #'train.kcsv'
        args.kcsv_val = os.path.join(myDatasetsPath, 'val.txt')
        args.kcsv_test = os.path.join(myDatasetsPath, 'test.txt')
        args.kcsv_classes = os.path.join(myDatasetsPath, "classes.txt")

        if args.run_script == 'Training':
            args.val_file = args.kcsv_val
        elif args.val_set == 'Val':
            args.val_file = args.kcsv_val
        else:
            args.val_file = args.kcsv_test

    elif args.dataset_name == "roots":
        args.train_csv_leaf_number_file = os.path.join(myDatasetsPath, 'sub_Train', "Train.csv")
        args.train_csv_leaf_location_file = os.path.join(myDatasetsPath, 'sub_Train','Train_pointsOutput.csv')
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(myDatasetsPath, "sub_" + args.val_set, args.val_set + ".csv")
        args.val_csv_leaf_location_file = os.path.join(myDatasetsPath, "sub_" + args.val_set,
                                                       args.val_set + "_pointsOutput.csv")
        args.val_json_file = None

        if args.dataset_type == "roots_json":
            args.train_json_file = os.path.join(myDatasetsPath, 'sub_Train', "Train_Dia_Length_Color.txt")
            args.val_json_file = os.path.join(myDatasetsPath, "sub_" + args.val_set, args.val_set + "_Dia_Length_Color.txt")

    if args.have_GT:
        args.base_dir = None
    else:
        args.base_dir = (
            os.path.join(myDatasetsPath, "sub_" + args.val_set)
            if args.dataset_name == "roots"
            else myDatasetsPath
        )

    if args.run_script == 'Inference':
        results_dir = config.General.experiment_path
        os.makedirs(results_dir, exist_ok=True)

        if args.to_draw:
            config.DrawProperties.save_img_path = os.path.join(results_dir, "Vis_" + args.val_set)
            os.makedirs(config.DrawProperties.save_img_path, exist_ok=True)

        config.General.files_path = os.path.join(results_dir, "OutputFiles_"+ args.val_set)

        args.txt_results = os.path.join(config.General.files_path,
                                        "with_Vis_results_"+ args.val_set +".txt" if config.General.to_draw else
                                        "without_Vis_results_"+ args.val_set +".txt")
        if config.DrawProperties.DRAW_MAPS:
            config.DrawProperties.maps_path = os.path.join(config.DrawProperties.save_img_path, "KP heatmaps")
            os.makedirs(config.DrawProperties.maps_path, exist_ok=True)

    else:
        config.General.files_path = os.path.join(config.General.experiment_path, "OutputFiles_Train")  # , 'Test2')

        args.txt_results = os.path.join(config.General.files_path, "Train_results.txt")

    os.makedirs(config.General.files_path, exist_ok=True)

    ######################################################################
    # weights dirs
    ######################################################################

    full_model_weights_dir = os.path.join(args.myExpPath, "Weights", 'full_model_weights')
    partial_weights_dir = os.path.join(args.myExpPath, "Weights", 'partial_weights')

    if args.weights_type == 'full_model_weights':
        weights_dir = full_model_weights_dir
    elif args.weights_type == 'partial_weights':
        weights_dir = partial_weights_dir

    if args.network_type in INCLUDE_BBOX_DETECTION:
        args.bbox_detection_weights_dir = os.path.join(args.myExpPath, "Weights", "bbox_detection")
        # if args.dataset_name == 'roots':
        #     args.bbox_detection_weights_dir = os.path.join(args.myExpPath, "Weights", "bbox_detection") #"bbox_detection") "bbox_detection_epoch69"
        # elif args.dataset_name == 'grapes':
        #     args.bbox_detection_weights_dir = os.path.join(args.myExpPath, 'per_object_counting', #'grapes_det_correct_filter_Score_0.7_nms_0.3',
        #                                                    "Weights", '2026-04-16_161416') #'2026-04-06_221233')

        os.makedirs(args.bbox_detection_weights_dir, exist_ok=True)

        if not args.network_type == "bbox_detection":

            if args.network_type == "per_object_counting":
                args.per_object_weights_dir = os.path.join(weights_dir, "per_object_counting") #args.myExpPath, "Weights"
                if args.estimate_type == 'withKeyPoints':
                    args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'counting_KP')
                elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'counting_Reg')

            elif args.network_type == "per_object_attributes":
                args.per_object_weights_dir = os.path.join(weights_dir, "per_object_attributes") #args.myExpPath, "Weights",
                if args.estimate_type == 'withKeyPoints':
                    args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'attributes_KP')
                elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'attributes_Reg')

            elif args.network_type == "per_object_attributes_multibranch":
                args.per_object_weights_dir = os.path.join(weights_dir, "per_object_attributes", 'both_Back2bFind2b') #args.myExpPath, "Weights",

            os.makedirs(args.per_object_weights_dir, exist_ok=True)

    if args.network_type == "per_image_estimation":
        if args.estimate_type == "withKeyPoints":
            args.per_image_weights_dir = os.path.join(full_model_weights_dir, "per_image_attributes", "TRL_KP") # args.myExpPath, "Weights",
        else:
            args.per_image_weights_dir = os.path.join(full_model_weights_dir, "per_image_attributes", "TRL_Reg") #args.myExpPath, "Weights", #os.path.join(args.myExpPath, 'TRL_estimator_reg\\Weights\\2026-05-21_121532') #os.path.join(args.myExpPath, "Weights", "per_image_attributes", "TRL_Reg")
        os.makedirs(args.per_image_weights_dir, exist_ok=True)

    # "D:\\from 16\\more_counting_Res\\more_counting_Res\\legonet_epoch=249.pt" #cont_legonet_epoch=14.pt"

    if args.load_bbox_det_weights:
        args.bbox_detection_weights_file = require_weights_file(
            args.bbox_weights_file,
            "--bbox-weights-file",
        )

    if args.load_per_object_counting_weights or args.load_per_object_attributes_weights:
        args.per_object_weights_file = require_weights_file(
            args.per_object_weights_file,
            "--per-object-weights-file",
        )

    if args.load_full_model_weights:
        args.full_model_weights = require_weights_file(
            args.full_weights_file,
            "--full-weights-file",
        )

    if args.save_from_model_file:
        file_path = "Prev_model_files\\"
        if args.dataset_name == "roots":
            if args.network_type == "per_image_estimation" and args.estimate_type == "withKeyPoints":
                file_path += "keyPoints_based_models\\TRL_only\\2023-01-23_132447"
                args.output_name = 'TRLwithKeyPoints'

            elif args.network_type == "per_image_estimation":
                file_path += "Reg_based_models", "reg_TRL_only\\2023-06-04_175138"
                args.output_name = 'TRLwithReg'

            elif args.network_type == "per_object_attributes":
                if args.estimate_type == 'withKeyPoints':
                    file_path += "keyPoints_based_models\\both_2_detect_3Sets\\2023-05-23_162315"
                    args.output_name = 'AttrWithKeyPoints'
                elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    file_path += "Reg_based_models\\both_2_detect_3Sets\\2023-05-28_194055"
                    args.output_name = 'AttrWithReg'

            elif args.network_type == "per_object_attributes_multibranch":
                args.output_name = 'AttrWith2B2F'
                #args.load_more_partial = True

                weights_dir_path = os.path.join(args.myExpPath, "Weights", "Prev_model_files")
                bbox_path = weights_dir_path + "\\Three_datasets_detection\\2023-05-20_230659\\legonet_epoch=69.pt"

                limit5Path = weights_dir_path + "\\keyPoints_based_models\\both_2_with detect\\limit5\\2023-02-28_184758\\legonet_epoch=231.pt" #\\saved_weights_epoch_231"
                # os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
                # "Results\\both_2_with detect", "changed_colorModel","2023-02-28_184758", "saved_weights_epoch_231")
                # should be in (path too long) : "I0.7_s 0.7_limit5_10dia_100color", #"fixed_detect", p
                # "2023-02-28_184758", "saved_weights_epoch_231")

                all_3setsPath = weights_dir_path + "\\keyPoints_based_models\\both_2_detect_3Sets\\2023-05-23_162315\\legonet_epoch=90.pt" #saved_weights_epoch_90"
                # os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
                #  "Results\\both_2_detect_3Sets\\I0.5_s0.7_10_dia_100_color", "2023-05-23_162315","saved_weights_epoch_90")
                args.model_path = {"bbox_path": bbox_path, "limit5Path": limit5Path, "all_3setsPath": all_3setsPath}

                args.additional_modules_weights = \
                    {"backbone_2": limit5Path,
                     "find_2": limit5Path,
                     "LeanCountingModule_color": limit5Path,
                     "LeanCountingModule_length": all_3setsPath,
                     "LeanCountingModule_diameter": limit5Path,
                     "find_2_b": os.path.join(all_3setsPath, "find_2"),
                     "backbone_2_b": os.path.join(all_3setsPath, "backbone_2")
                     }

                # args.additional_modules_weights["find_2_b"] = os.path.join(all_3setsPath, "find_2")
                # args.additional_modules_weights["backbone_2_b"] = os.path.join(all_3setsPath, "backbone_2")

            elif args.network_type == "bbox_detection":
                file_path += "Three_datasets_detection\\2023-05-20_230659"


        elif args.dataset_name == "grapes":
            if args.network_type == "per_object_counting":
                if args.estimate_type == 'withKeyPoints':
                    file_path += "both_old" #"per_object_counting_trained\\Weights\\2026-04-16_161416"
                    args.output_name = 'CountWithKeyPoints'
                elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    file_path =  ""

        if not args.network_type == "per_object_attributes_multibranch":
            args.weights_dir = os.path.join(args.myExpPath, "Weights", file_path)
            args.model_path = get_weights_file(args.weights_dir)  # for initial load of old weights file


    #"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults (1)\\KK_Exp_Results_last\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
    # #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')
    #os.path.join(results_dir,'grapes_twoBack_reg_P3_P5_fluid_new', 'saved_weights_164')

    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run a LegoNet training or inference experiment."""
    warnings.filterwarnings("ignore")
    faulthandler.enable()
    start_time = time.time()

    try:
        args = configure_runtime(parse_args(argv))
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    from legonet import runner

    runner.run(args)

    execution_time = time.time() - start_time
    finalize_text_results(args, execution_time)
    print(f"Execution time in minutes: {execution_time / 60:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
