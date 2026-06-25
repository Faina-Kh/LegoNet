import os
import sys
import argparse
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import paths
import numpy as np
import csv
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

startTime = time.time()


import faulthandler
faulthandler.enable()

time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

########################################################################################################################

def print_to_csv(args, executionTime):
    if getattr(args, "txt_results", ""):
        with open(args.txt_results, 'a') as f:
            f.write(f'Execution time in minutes: {(executionTime / 60):.2f}\n')

        with open(args.txt_results, "r", encoding="utf-8") as f_in, \
                open(args.output_csv, "w", newline="", encoding="utf-8") as f_out:

            writer = csv.writer(f_out)
            for line in f_in:
                writer.writerow(line.strip().split("|"))


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


def parse_args(args):
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
            "counting_lean",
            "counting_reg",
            "both",
            "both_for_roots_2",
            "both_Back2bFind2b",
        ],
        default=None,
    )
    parser.add_argument("--current-results-dir", "--current_results_dir", default=None)
    parser.add_argument(
        "--estimate-type",
        "--estimate_type",
        choices=["withKeyPoints", "reg_fpn_p3_p7_min_sig"],
        default=None,
    )
    parser.add_argument("--run-script", "--run_script", choices=["Training", "Inference"], default=None)
    parser.add_argument("--val-set", "--val_set", choices=["Val", "Test"], default=None)
    parser.add_argument("--num-of-epochs", "--num_of_epochs", type=int, default=None)
    parser.add_argument("--have-gt", "--have_GT", type=parse_bool, default=None)
    parser.add_argument("--to-draw", "--to_draw", type=parse_bool, default=None)
    parser.add_argument("--save-from-model-file", "--save_from_model_file", type=parse_bool, default=None)
    parser.add_argument("--load-weights", "--load_weights", type=parse_bool, default=None)
    parser.add_argument("--evaluate-detection", "--evaluate_detection", type=parse_bool, default=None)
    parser.add_argument("--weights-type", "--weights_type", choices=['full_model_weights', 'partial_weights'], default=None)

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


# parse arguments
args = None

if args is None:
    args = sys.argv[1:]
    args = parse_args(args)

DEFAULT_ESTIMATE_TYPE_BY_NETWORK = {
    "counting_lean": "withKeyPoints",
    "counting_reg": "reg_fpn_p3_p7_min_sig",
    "both_Back2bFind2b": "withKeyPoints",
    #"both": "withKeyPoints" # dont have currently weights for "reg_fpn_p3_p7_min_sig"
}

MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS = ("bbox_detection")

BOTH_NETWORKS = ("both", "both_for_roots_2", "both_Back2bFind2b")

INCLUDE_BBOX_DETECTION = ("bbox_detection", "both", "both_for_roots_2", "both_Back2bFind2b")


NETWORKS_OPTIONS_BY_DATASETS = {'roots': ("bbox_detection", "counting_lean", "counting_reg", "both_for_roots_2",
                                          "both_Back2bFind2b"),
                                'grapes': ("bbox_detection", "both")
                                }
########################################################################################################################
# user definitions
########################################################################################################################
args.gpu_num = args.gpu_num or '1'

args.STORAGE_PATH = args.storage_path or 'C:\\Users\\bordezki\\Desktop\\LegoNet' #'C:\\Users\\borde\\Desktop\\Faina_code\\LegoNet' #'C:\\Users\\bordezki\\Desktop\\LegoNet' #'C:\\Users\\borde\\Desktop\\פאינה\\LegoNet' #r"C:\Users\bordezki\Desktop\LegoNet"
args.dataset_name = args.dataset_name or "grapes" #"grapes" #"roots"
args.network_type = args.network_type or "both" # "counting_lean"  #"counting_reg" #"both_Back2bFind2b" #"both_for_roots_2" # "both" #"bbox_detection"
args.estimate_type = args.estimate_type or "reg_fpn_p3_p7_min_sig" #DEFAULT_ESTIMATE_TYPE_BY_NETWORK.get(args.network_type, "reg_fpn_p3_p7_min_sig")
args.to_draw = args.to_draw or False

args.run_script = args.run_script or 'Training' #'Training' #'Inference'
args.val_set = args.val_set or "Val" #"Test" #"Val"
args.have_GT = args.have_gt or True

args.num_of_epochs = args.num_of_epochs or 300
type_name = '_KP_' if args.estimate_type == "withKeyPoints" else "_Reg_"

if args.run_script == 'Training':
    args.current_results_dir = args.current_results_dir or (args.network_type + type_name + 'Training'+ '_check_IoUavg_score_0.05') #'_epoch_by_IoUavg') #+'_gt count by points')
else:
    args.current_results_dir = args.current_results_dir or (args.network_type + type_name + args.val_set + 'epoch_by_IoUavg_'+ 'new_84' ) #'_'+time_stemp

#--------------------------------------------------------------------------------------------------------------
args.choose_epoch_by_IoUavg = True
config.Detection.min_score = 0.05 # not 0.7

args.evaluate_detection = False #args.evaluate_detection or args.network_type in MANDATORY_DETECTION_EVAL_NETWORK_OPTIONS

args.load_only_bbox_weights = False

#--------------------------------------------------------------------------------------------------------------
args.evaluate_both = args.network_type in BOTH_NETWORKS

args.load_weights = args.load_weights or True

args.save_from_model_file = not args.load_weights # args.save_from_model_file or True

args.weights_type = args.weights_type or 'partial_weights' # 'full_model_weights'  #'partial_weights'
assert args.weights_type == 'full_model_weights' or args.weights_type == 'partial_weights'

if args.weights_type == 'full_model_weights':
    args.load_full_model_weights = True
elif args.weights_type == 'partial_weights':
    args.load_full_model_weights = False

args.load_partial_weights = not args.load_full_model_weights

## To add as options for the user or not?
args.load_bbox_det_weights = True
# args.load_per_object_counting_weights = True
# args.load_per_object_attributes_weights = True


if args.load_only_bbox_weights:
    args.load_per_object_counting_weights = False
    args.load_per_object_attributes_weights = False

elif args.load_partial_weights:
    if args.network_type == "both":
        args.load_per_object_counting_weights = True
    else:
        args.load_per_object_counting_weights = False

    args.load_per_object_attributes_weights = not args.load_per_object_counting_weights



########################################################################################################################
# GPU settings
########################################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'Running on physical GPU {args.gpu_num}')
else:
    device = torch.device("cpu")

config.General.device = device

######################################################################
# General paths
######################################################################

myPaths = paths.get_paths(args.STORAGE_PATH, args.dataset_name)
myDatasetsPath = myPaths["DATASETS_PATH"]
args.myExpPath = myPaths["EXP_RESULTS_PATH"]

config.General.experiment_path = os.path.join(args.myExpPath, 'Results', args.current_results_dir)
os.makedirs(config.General.experiment_path, exist_ok=True)

config.General.dataset_name= args.dataset_name
config.General.MODE = args.run_script
config.General.model_name = args.network_type
config.Detect_and_Estimate.type = args.network_type

config.AttributeEstimation.calc_det_performance = args.evaluate_detection

#"grapes diff radii"
#"grapes_twoBack_keyP_sameRadii_train perfect det_fancy aug_20Per_test perfect det"
#"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_40Per"
#'grapes_twoBack_keyP_different radii_fixed' #"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_40Per" #"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_30Per"
#"grapes_reg P3_P7_crop from P3_fluid"
#"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_20Per"
#"grapes_twoBack_reg_P3_P5_fixed with better detector"
#"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_20Per" #"grapes_twoBack_reg_P3_P5_fixed with better detector"
#"grapes_twoBack_keyP_sameRadii_perfect det_fancy aug" #"grapes_twoBack_keyP_sameRadii_perfect det_fixed aug"
#"grapes_twoBack_reg_checkGrads"
#grapes_twoBack_reg_P3_P5_fluid_new" #"grapes_twoBack_keypoints_sameRadi_fluid_new"
#"grapes_twoBack_reg_P3_P5_fluid_new"
#"grapes_twoBack_keypoints_sameRadi_fluid_new"
#"grapes_twoBack_keypoints_sameRadi_fluid" #"grapes_twoBack_reg_P3_P5_fluid"



# True if loading pre-trained weights


# task specific definitions - ToDo - organize per task type
args.freeze_detection = True
args.eval_in_train = True

args.eval_detection_params = False

if args.network_type != "bbox_detection":
    args.freeze_detection = True

if args.dataset_name == 'grapes':
    #if args.network_type == "bbox_detection":
    args.filter_empty_bbox = True
    config.General.filter_empty_bbox = args.filter_empty_bbox

    config.General.predict_empty_image = False

    config.AttributeEstimation.do_nmcs = True

elif args.dataset_name == 'roots':

    if args.network_type in INCLUDE_BBOX_DETECTION: #args.network_type == "bbox_detection" or args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
        config.Detection.change_anchors = True
        config.Detection.ratios = np.array([0.5, 1, 3])  # np.array([0.5, 1, 3]) #np.array([0.5, 1, 4]) #np.array([0.5, 1, 2])
        print("Detection.ratios:", config.Detection.ratios)

        #config.General.predict_empty_image = False

        config.AttributeEstimation.do_nmcs = False
        args.dia_loss_weight = 10  # 10 #1000 #10 #100
        args.color_loss_weight = 100  # 100
        args.maps_loss_weight = 1 #10

        # #os.path.join(args.dataset_path,"Results\\detection\\IOU 0.7_score 0.7_limit_5\\2023-02-15_154227","saved_weights_epoch_280")
        # "IOU 0.9_score 0.7_limit_5\\2023-02-16_150507" ,"saved_weights_epoch_158")
        # "Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", " saved_weights_epoch_175")
        # "C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults\\KK_Exp_Results\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
        # #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')

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

######################################################################
# Checks
######################################################################
assert args.dataset_name == "grapes" or args.dataset_name == "roots"
assert (args.network_type == "bbox_detection" or args.network_type == "counting_lean" or
        args.network_type == "counting_reg" or args.network_type == "both" or args.network_type == "both_for_roots_2"
        or args.network_type == "both_Back2bFind2b" )


# ToDo- replace args.network_type = "both" or args.network_type = "both_for_roots_2" with 'perObjectEstimate
# estimate_type = 'counting' , 'TRL',
# config.Estimate.estimate_type


assert args.val_set == 'Val' or args.val_set == 'Test'

assert (config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig' or
        config.AttributeEstimation.estimate_type == 'withKeyPoints')

assert args.run_script =='Training' or args.run_script =='Inference'

if args.run_script == 'Training':
    assert args.have_GT == True
    assert args.val_set == "Val"
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
    if args.network_type == "counting_lean" or args.network_type == "counting_reg":
        args.dataset_type = 'csv_LCC'
    else:
        args.dataset_type = "roots_json"

if args.network_type == "bbox_detection":
    config.General.NETWORK_TYPE = config.NetworkType.detection
elif args.network_type == "counting_lean":
    config.General.NETWORK_TYPE = config.NetworkType.counting_lean
elif args.network_type == "counting_reg":
    config.General.NETWORK_TYPE = config.NetworkType.counting_reg
elif args.network_type == "both" or args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
    config.General.NETWORK_TYPE = config.NetworkType.detection_and_counting

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
    args.base_dir = myDatasetsPath

if args.run_script == 'Inference':
    if args.network_type == 'bbox_detection':
        results_dir = os.path.join(config.General.experiment_path, 'Inf_min_score_'+str(config.Detection.min_score))
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = os.path.join(config.General.experiment_path)

    if args.to_draw:
        config.DrawProperties.save_img_path = os.path.join(results_dir, "Vis_" + args.val_set)
        os.makedirs(config.DrawProperties.save_img_path, exist_ok=True)

    config.General.files_path = os.path.join(results_dir, "OutputFiles_"+ args.val_set) #, 'Test2')

    args.txt_results = os.path.join(config.General.files_path,
                                    "with_Vis_results_"+ args.val_set +".txt" if config.General.to_draw else
                                    "without_Vis_results_"+ args.val_set +".txt")
    args.output_csv = os.path.join(config.General.files_path,
                                   "with_Vis_results_"+ args.val_set +".csv" if config.General.to_draw else
                                   "without_Vis_results_"+ args.val_set +".csv")

    if config.DrawProperties.DRAW_MAPS:
        config.DrawProperties.maps_path = os.path.join(config.DrawProperties.save_img_path, "points_maps")
        os.makedirs(config.DrawProperties.maps_path, exist_ok=True)

else:
    config.General.files_path = os.path.join(config.General.experiment_path, "OutputFiles_Train")  # , 'Test2')

    args.txt_results = os.path.join(config.General.files_path, "Train_results.txt")
    args.output_csv = os.path.join(config.General.files_path, "Train_results.csv")

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

        if args.network_type == "both":
            args.per_object_weights_dir = os.path.join(weights_dir, "per_object_counting") #args.myExpPath, "Weights"
            if args.estimate_type == 'withKeyPoints':
                args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'counting_KP')
            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'counting_Reg')

        elif args.network_type == "both_for_roots_2":
            args.per_object_weights_dir = os.path.join(weights_dir, "per_object_attributes") #args.myExpPath, "Weights",
            if args.estimate_type == 'withKeyPoints':
                args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'attributes_KP')
            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                args.per_object_weights_dir = os.path.join(args.per_object_weights_dir, 'attributes_Reg')

        elif args.network_type == "both_Back2bFind2b":
            args.per_object_weights_dir = os.path.join(weights_dir, "per_object_attributes", 'both_Back2bFind2b') #args.myExpPath, "Weights",

        os.makedirs(args.per_object_weights_dir, exist_ok=True)


if args.network_type == "counting_lean" or args.network_type == "counting_reg":
    if args.network_type == "counting_lean":
        args.per_image_weights_dir = os.path.join(full_model_weights_dir, "per_image_attributes", "TRL_KP") # args.myExpPath, "Weights",
    else:
        args.per_image_weights_dir = os.path.join(full_model_weights_dir, "per_image_attributes", "TRL_Reg") #args.myExpPath, "Weights", #os.path.join(args.myExpPath, 'TRL_estimator_reg\\Weights\\2026-05-21_121532') #os.path.join(args.myExpPath, "Weights", "per_image_attributes", "TRL_Reg")
    os.makedirs(args.per_image_weights_dir, exist_ok=True)

# "D:\\from 16\\more_counting_Res\\more_counting_Res\\legonet_epoch=249.pt" #cont_legonet_epoch=14.pt"

if args.load_bbox_det_weights:
    args.bbox_detection_weights_file = get_weights_file(args.bbox_detection_weights_dir)
    # args.partial_weights_dir = os.path.join(args.myExpPath, "Weights",
    #                                         "Prev_model_files\\Three_datasets_detection\\2023-05-20_230659",
    #                                         "saved_weights_epoch_69")

if args.load_per_object_counting_weights or args.load_per_object_attributes_weights:
    args.per_object_weights_file = get_weights_file(args.per_object_weights_dir)

# if args.load_per_image_weights:
#     args.per_image_weights_file = get_weights_file(args.per_image_weights_dir)
if args.load_full_model_weights:
    if args.network_type == "counting_lean" or args.network_type == "counting_reg":
        args.full_model_weights = get_weights_file(args.per_image_weights_dir)
    else:
        args.full_model_weights = get_weights_file(args.per_object_weights_dir) #args.bbox_detection_weights_dir)

if args.save_from_model_file:
    file_path = "Prev_model_files\\"
    if args.dataset_name == "roots":
        if args.network_type == "counting_lean":
            file_path += "keyPoints_based_models\\TRL_only\\2023-01-23_132447"
            args.output_name = 'TRLwithKeyPoints'

        elif args.network_type == "counting_reg":
            file_path += "Reg_based_models", "reg_TRL_only\\2023-06-04_175138"
            args.output_name = 'TRLwithReg'

        elif args.network_type == "both_for_roots_2":
            if args.estimate_type == 'withKeyPoints':
                file_path += "keyPoints_based_models\\both_2_detect_3Sets\\2023-05-23_162315"
                args.output_name = 'AttrWithKeyPoints'
            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                file_path += "Reg_based_models\\both_2_detect_3Sets\\2023-05-28_194055"
                args.output_name = 'AttrWithReg'

        elif args.network_type == "both_Back2bFind2b":
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
        if args.network_type == "both":
            if args.estimate_type == 'withKeyPoints':
                file_path += "both_old" #"per_object_counting_trained\\Weights\\2026-04-16_161416"
                args.output_name = 'CountWithKeyPoints'
            elif args.estimate_type == 'reg_fpn_p3_p7_min_sig':
                file_path =  ""

    if not args.network_type == "both_Back2bFind2b":
        args.weights_dir = os.path.join(args.myExpPath, "Weights", file_path)
        args.model_path = get_weights_file(args.weights_dir)  # for initial load of old weights file


#"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults (1)\\KK_Exp_Results_last\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
# #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')
#os.path.join(results_dir,'grapes_twoBack_reg_P3_P5_fluid_new', 'saved_weights_164')


########################################################################################################################
# Run the code
########################################################################################################################

# Run training or validation of a specific model
import legonet.runner

legonet.runner.run(args)

# run validation on multiple models with different min_score and iou_threshold
#legonet.runner.run_offline_validation(args)

#legonet.runner.visualize_detection(args)

########################################################################################################################

executionTime = (time.time() - startTime)
print(f'Execution time in minutes: {(executionTime/60):.3f}')

print_to_csv(args, executionTime)
########################################################################################################################
