import os
import sys
import argparse
import torch
import time
import config
import paths
import numpy as np
from pathlib import Path
import csv
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import legonet.runner

startTime = time.time()

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


def parse_args(args):
    parser = argparse.ArgumentParser(description='main script.')

    return parser.parse_args(args)


# parse arguments
args = None

if args is None:
    args = sys.argv[1:]
    args = parse_args(args)


########################################################################################################################
# user definitions
########################################################################################################################
args.gpu_num = '0'

args.STORAGE_PATH = 'C:\\Users\\bordezki\\Desktop\\LegoNet' #'C:\\Users\\borde\\Desktop\\פאינה\\LegoNet' #r"C:\Users\bordezki\Desktop\LegoNet"
args.dataset_name = "grapes" #"roots"
args.network_type = "both" # "both" #"bbox_detection"
args.weights_dir = "" # currently not relevant

config.Detection.min_score = 0.7 #0.7 #0.05
args.current_results_dir = 'per_object_counting' #'bbox_detection' # 'per_object_counting'

#args.network_type + '\\Inf_min_score_0.7' #"grapes_detection_minScore 0.05" #"grapes_detection_filterBBOX_minScore 0.7"

args.run_script = 'Training' #'Training' #'Inference'
config.General.MODE = args.run_script

config.General.model_name = args.network_type

val_set = "Val"

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
args.load_weights = True
args.load_partial_weights = True

args.load_bbox_det_weights = True
args.load_per_object_counting_weights = False

args.load_more_partial = False

args.have_GT = True
args.num_of_epochs = 300
args.do_Kfold = False

# task specific definitions - ToDo - organize per task type
args.freeze_detection = True
args.eval_in_train = True

args.evaluate_detection = False

args.eval_detection_params = False
args.evaluate_both = True # relevant to validation, not train (roots)

config.AttributeEstimation.estimate_type = 'withKeyPoints' #'reg_fpn_p3_p7_min_sig'

config.Detect_and_Estimate.two_backbones = True


if (args.network_type == "bbox_detection" and args.dataset_name == 'roots') or args.network_type == "both_for_roots_2":
    config.Detection.change_anchors = True
    config.Detection.ratios = np.array([0.5, 1, 3])  # np.array([0.5, 1, 3]) #np.array([0.5, 1, 4]) #np.array([0.5, 1, 2])
    print("Detection.ratios:", config.Detection.ratios)


if args.network_type != "bbox_detection":
    args.freeze_detection = True

if args.dataset_name == 'grapes':
    if args.network_type == "bbox_detection":
        args.filter_empty_bbox = True
        config.General.filter_empty_bbox = args.filter_empty_bbox

    config.AttributeEstimation.do_nmcs = True
    config.Detection.NMS_THRESHOLD = 0.3  # default was 0.5 in config

elif args.dataset_name == 'roots':
    config.AttributeEstimation.do_nmcs = False
    config.General.predict_empty_image = True #why?


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
# ToDo - to resolve
######################################################################

args.num_workers = 0 # 0 - for single processing
args.batch_size = 1
args.output_size = 1

args.pre_process = 'torch_like' #'keras_like'  # torch_like
args.backbone_type = "ResNetBackboneModule"

args.loss_weight = 1  # 1  #1000 #10 #100 # roots_both ablations

config.General.binary_model = False
config.General.binary_version = ""

##########################################################################################################
# ToDo - remove
config.General.with_new_layers = False  # should be False only for points TRL
config.General.with_new_layers_for_both = False # always False for TRL_both, True for regular TRL
##########################################################################################################

######################################################################
# Checks
######################################################################
assert args.dataset_name == "grapes" or args.dataset_name == "roots"
assert (args.network_type == "bbox_detection" or args.network_type == "counting_lean" or
        args.network_type == "counting_reg" or args.network_type == "both" or args.network_type == "both_for_roots_2")


# ToDo- replace args.network_type = "both" or args.network_type = "both_for_roots_2" with 'perObjectEstimate
# estimate_type = 'counting' , 'TRL',
# config.Estimate.estimate_type


assert val_set == 'Val' or val_set == 'Test'

assert config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig' or config.AttributeEstimation.estimate_type == 'withKeyPoints'

assert args.run_script =='Training' or args.run_script =='Inference'
if args.run_script == 'Training':
    assert args.have_GT == True
    args.epochs = args.num_of_epochs
    val_set = "Val"
    config.General.to_draw = False
    config.DrawProperties.DRAW_MAPS = False
    config.AttributeEstimation.calc_det_performance = False
else:
    config.General.to_draw = True
    config.DrawProperties.DRAW_MAPS = True
    config.AttributeEstimation.calc_det_performance = True


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
else:
    args.dataset_type = "coco"
    args.dataset_name = 'tomato_fruit_12_3_18'

if args.network_type == "bbox_detection":
    config.General.NETWORK_TYPE = config.NetworkType.detection
elif args.network_type == "counting_lean":
    config.General.NETWORK_TYPE = config.NetworkType.counting_lean
elif args.network_type == "counting_reg":
    config.General.NETWORK_TYPE = config.NetworkType.counting_reg
elif args.network_type == "both" or args.network_type == "both_for_roots_2":
    config.General.NETWORK_TYPE = config.NetworkType.detection_and_counting


######################################################################
# Paths
######################################################################
config.General.dataset_name= args.dataset_name

myPaths = paths.get_paths(args.STORAGE_PATH, args.dataset_name)
myDatasetsPath = myPaths["DATASETS_PATH"]
args.myExpPath = myPaths["EXP_RESULTS_PATH"]

config.General.experiment_path = os.path.join(args.myExpPath, args.current_results_dir)
os.makedirs(config.General.experiment_path, exist_ok=True)

config.General.weights_dir = os.path.join(config.General.experiment_path, "Weights")
if args.run_script == 'Training':
    time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    config.General.weights_dir += "\\"+time_stemp
    os.makedirs(config.General.weights_dir, exist_ok=True)
else:
    config.General.weights_dir = args.weights_dir # user defined path

########################################################################################################################
# Define the paths
########################################################################################################################

if args.dataset_type == 'kcsv':
    args.kcsv_train = os.path.join(myDatasetsPath, 'train.kcsv')
    args.kcsv_val = os.path.join(myDatasetsPath, 'val.kcsv')
    args.kcsv_test = os.path.join(myDatasetsPath, 'test.kcsv')
    args.kcsv_classes = os.path.join(myDatasetsPath, "classes.kcsv")

    if args.run_script == 'Training':
        args.val_file = args.kcsv_val
    elif val_set == 'Val':
        args.val_file = args.kcsv_val
    else:
        args.val_file = args.kcsv_test

elif args.dataset_name == "roots":
    #sample_n = "200"

    if args.do_Kfold:
        k = "1"
        args.train_csv_leaf_number_file = os.path.join(args.dataset_path,'sub_Train_'+k, "Train.csv")
        args.train_csv_leaf_location_file = os.path.join(args.dataset_path,'sub_Train_'+k, 'Train_pointsOutput.csv')
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(args.dataset_path,"sub_"+val_set +"_"+k,  val_set+".csv" )
        args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "sub_"+val_set +"_"+k,
                                                       val_set+"_pointsOutput.csv")
        args.val_json_file = None

        if args.dataset_type == "roots_json":
            args.train_json_file = os.path.join(args.dataset_path, 'sub_Train_' + k, "Train_Dia_Length_Color.txt")
            args.val_json_file = os.path.join(args.dataset_path, "sub_" + val_set + "_" + k,
                                              val_set + "_Dia_Length_Color.txt")
    else:
        args.train_csv_leaf_number_file = os.path.join(args.dataset_path, 'sub_Train', "Train.csv")
        args.train_csv_leaf_location_file = os.path.join(args.dataset_path, 'sub_Train','Train_pointsOutput.csv')
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(args.dataset_path, "sub_" + val_set, val_set + ".csv")
        args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "sub_" + val_set,
                                                       val_set + "_pointsOutput.csv")
        args.val_json_file = None

        if args.dataset_type == "roots_json":
            args.train_json_file = os.path.join(args.dataset_path, 'sub_Train', "Train_Dia_Length_Color.txt")
            args.val_json_file = os.path.join(args.dataset_path, "sub_" + val_set, val_set + "_Dia_Length_Color.txt")

elif args.dataset_type == "csv_LCC" and args.dataset_name != "roots":

    ds = 'A4'
    args.dataset_name = 'Counting Datasets\\CVPPP2017_LCC_training\\training\\' + ds
    args.train_csv_leaf_number_file = os.path.join(myDatasetsPath, args.dataset_name, 'train', ds + '_Train.csv')
    args.train_csv_leaf_location_file = os.path.join(myDatasetsPath, args.dataset_name, 'train',
                                                     ds + '_Train_leaf_location.csv')
    args.val_csv_leaf_number_file = os.path.join(myDatasetsPath, args.dataset_name, 'val', ds + '_Val.csv')
    args.val_csv_leaf_location_file = os.path.join(myDatasetsPath, args.dataset_name, 'val',
                                                   ds + '_Val_leaf_location.csv')


if args.have_GT:
    args.base_dir = None
else:
    args.base_dir = args.dataset_path

if args.run_script == 'Inference':
    if args.network_type == 'bbox_detection':
        results_dir = os.path.join(config.General.experiment_path, 'Inf_min_score_'+str(config.Detection.min_score))
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = config.General.experiment_path
    args.txt_results = os.path.join(results_dir,
                                    "with_Vis_results_"+val_set +".txt" if config.General.to_draw else
                                    "without_Vis_results_"+val_set +".txt")
    args.output_csv = os.path.join(results_dir,
                                   "with_Vis_results_"+ val_set +".csv" if config.General.to_draw else
                                   "without_Vis_results_"+ val_set +".csv")

    config.DrawProperties.save_img_path = os.path.join(config.General.experiment_path, "Vis_" + val_set)
    os.makedirs(config.DrawProperties.save_img_path, exist_ok=True)

    if config.DrawProperties.DRAW_MAPS:
        config.DrawProperties.maps_path = os.path.join(config.DrawProperties.save_img_path, "points_maps")
        os.makedirs(config.DrawProperties.maps_path, exist_ok=True)

else:
    args.txt_results = os.path.join(config.General.experiment_path, "Train_results.txt")
    args.output_csv = os.path.join(config.General.experiment_path, "Train_results.csv")


######################################################################
# weights related
######################################################################

if args.load_weights:
    args.bbox_detection_weights_dir = os.path.join(args.myExpPath, "Weights", "bbox_detection")
    args.per_object_weights_dir = os.path.join(args.myExpPath, "Weights", "per_object_counting")
    args.model_path = os.path.join(args.myExpPath, "Weights", "both", "legonet_epoch=249.pt")
    os.makedirs(args.bbox_detection_weights_dir, exist_ok=True)
    os.makedirs(args.per_object_weights_dir, exist_ok=True)
    # "D:\\from 16\\more_counting_Res\\more_counting_Res\\legonet_epoch=249.pt" #cont_legonet_epoch=14.pt"

    if args.load_bbox_det_weights:
        files = [p for p in Path(args.bbox_detection_weights_dir).iterdir() if p.is_file()]
        if len(files)>0:
            args.bbox_detection_weights_file = str(files[0])

    if args.load_per_object_counting_weights:
        files = [p for p in Path(args.per_object_weights_dir).iterdir() if p.is_file()]
        if len(files) > 0:
            args.per_object_weights_file = str(files[0])

else:
    args.weights_file_path = ""

args.partial_weights_dir = ""
#"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults (1)\\KK_Exp_Results_last\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
# #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')
#os.path.join(results_dir,'grapes_twoBack_reg_P3_P5_fluid_new', 'saved_weights_164')


########################################################################################################################
# Run the code
########################################################################################################################

# Run training or validation of a specific model
legonet.runner.run(args)

# run validation on multiple models with different min_score and iou_threshold
#legonet.runner.run_offline_validation(args)

#legonet.runner.visualize_detection(args)

########################################################################################################################

executionTime = (time.time() - startTime)
print(f'Execution time in minutes: {(executionTime/60):.3f}')

print_to_csv(args, executionTime)
########################################################################################################################