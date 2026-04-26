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

def get_weights_file(weights_dir):
    files = [p for p in Path(weights_dir).iterdir() if p.is_file()]
    if len(files) > 0:
        return str(files[0])


# parse arguments
args = None

if args is None:
    args = sys.argv[1:]
    args = parse_args(args)


########################################################################################################################
# user definitions
########################################################################################################################
args.gpu_num = '0'

args.STORAGE_PATH = 'C:\\Users\\borde\\Desktop\\Faina_code\\LegoNet' #'C:\\Users\\bordezki\\Desktop\\LegoNet' #'C:\\Users\\borde\\Desktop\\פאינה\\LegoNet' #r"C:\Users\bordezki\Desktop\LegoNet"
args.dataset_name = "roots" #"grapes" #"roots"
args.network_type = "both_for_roots_2" # "both" #"bbox_detection"
args.current_results_dir = 'per_object_attributes' #'per_object_counting' #'bbox_detection' # 'per_object_counting'
args.run_script = 'Inference' #'Training' #'Inference'
val_set = "Val"

######################################################################
# General paths
######################################################################

myPaths = paths.get_paths(args.STORAGE_PATH, args.dataset_name)
myDatasetsPath = myPaths["DATASETS_PATH"]
args.myExpPath = myPaths["EXP_RESULTS_PATH"]

config.General.experiment_path = os.path.join(args.myExpPath, args.current_results_dir)
os.makedirs(config.General.experiment_path, exist_ok=True)

config.General.dataset_name= args.dataset_name
#args.network_type + '\\Inf_min_score_0.7' #"grapes_detection_minScore 0.05" #"grapes_detection_filterBBOX_minScore 0.7"
config.General.MODE = args.run_script
config.General.model_name = args.network_type
config.Detect_and_Estimate.type = args.network_type

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
args.load_per_object_attributes_weights = True
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

if args.network_type == 'bbox_detection' or args.network_type == "both" or args.network_type == "both_for_roots_2":
    config.Detection.min_score = 0.7  # 0.7 #0.05


if args.dataset_name == 'grapes':
    if args.network_type == "bbox_detection":
        args.filter_empty_bbox = True
        config.General.filter_empty_bbox = args.filter_empty_bbox

    config.AttributeEstimation.do_nmcs = True
    config.Detection.NMS_THRESHOLD = 0.3  # default was 0.5 in config

elif args.dataset_name == 'roots':

    if args.network_type == 'bbox_detection' or args.network_type == "both_for_roots_2":
        config.Detection.iou_threshold = 0.5 #0.7 #0.9  # 0.5 #[0.3, 0.5, 0.7, 0.9, 0.95] #0.5
        config.Detection.min_score = 0.7 # 0.8  #0.7 #0.95 #0.5 #0.05

    if args.network_type == "both_for_roots_2":
        config.General.predict_empty_image = True  # why?
        config.AttributeEstimation.do_nmcs = False
        args.dia_loss_weight = 10  # 10 #1000 #10 #100
        args.color_loss_weight = 100  # 100
        args.maps_loss_weight = 1 #10
        config.General.binary_loss_version = "L1Loss"
        #################################################################
        #config.Detect_and_Estimate.use_new_Find = False
        #config.General.with_new_layers_for_both = False  # always False for TRL_both, True for regular TRL

        args.load_more_partial = False
        limit5Path = ""  # os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
        # "Results\\both_2_with detect", "changed_colorModel","2023-02-28_184758", "saved_weights_epoch_231")
        # should be in (path too long) : "I0.7_s 0.7_limit5_10dia_100color", #"fixed_detect", p
        # "2023-02-28_184758", "saved_weights_epoch_231")

        all_3setsPath = ""  # os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
        #  "Results\\both_2_detect_3Sets\\I0.5_s0.7_10_dia_100_color", "2023-05-23_162315","saved_weights_epoch_90")

        args.additional_modules_weights = ""
        # {"backbone_2":limit5Path,
        #  "find_2": limit5Path,
        #  "LeanCountingModule_color": limit5Path,
        #  "LeanCountingModule_length": all_3setsPath,
        #  "LeanCountingModule_diameter": limit5Path}

        config.General.twoFind_2 = False
        config.General.twoBackbone_2 = False

        if config.General.twoFind_2:
            args.additional_modules_weights["find_2_b"] = os.path.join(all_3setsPath, "find_2")
        if config.General.twoBackbone_2:
            args.additional_modules_weights["backbone_2_b"] = os.path.join(all_3setsPath, "backbone_2")

        # #os.path.join(args.dataset_path,"Results\\detection\\IOU 0.7_score 0.7_limit_5\\2023-02-15_154227","saved_weights_epoch_280")
        # "IOU 0.9_score 0.7_limit_5\\2023-02-16_150507" ,"saved_weights_epoch_158")
        # "Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", " saved_weights_epoch_175")
        # "C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults\\KK_Exp_Results\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
        # #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')

        #################################################################




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

#config.General.binary_model = False


##########################################################################################################
# ToDo - remove
config.General.with_new_layers = False  # should be False only for points TRL

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

assert (config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig' or
        config.AttributeEstimation.estimate_type == 'withKeyPoints')

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



########################################################################################################################
# Define the paths
########################################################################################################################
if args.run_script == 'Training':
    config.General.weights_dir = os.path.join(config.General.experiment_path, "Weights")
    time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    config.General.weights_dir += "\\"+time_stemp
    os.makedirs(config.General.weights_dir, exist_ok=True)
# else:
#     config.General.weights_dir = args.weights_dir # user defined path


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
        args.train_csv_leaf_number_file = os.path.join(myDatasetsPath,'sub_Train_'+k, "Train.csv")
        args.train_csv_leaf_location_file = os.path.join(myDatasetsPath,'sub_Train_'+k, 'Train_pointsOutput.csv')
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(myDatasetsPath,"sub_"+val_set +"_"+k,  val_set+".csv" )
        args.val_csv_leaf_location_file = os.path.join(myDatasetsPath, "sub_"+val_set +"_"+k,
                                                       val_set+"_pointsOutput.csv")
        args.val_json_file = None

        if args.dataset_type == "roots_json":
            args.train_json_file = os.path.join(myDatasetsPath, 'sub_Train_' + k, "Train_Dia_Length_Color.txt")
            args.val_json_file = os.path.join(myDatasetsPath, "sub_" + val_set + "_" + k,
                                              val_set + "_Dia_Length_Color.txt")
    else:
        args.train_csv_leaf_number_file = os.path.join(myDatasetsPath, 'sub_Train', "Train.csv")
        args.train_csv_leaf_location_file = os.path.join(myDatasetsPath, 'sub_Train','Train_pointsOutput.csv')
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(myDatasetsPath, "sub_" + val_set, val_set + ".csv")
        args.val_csv_leaf_location_file = os.path.join(myDatasetsPath, "sub_" + val_set,
                                                       val_set + "_pointsOutput.csv")
        args.val_json_file = None

        if args.dataset_type == "roots_json":
            args.train_json_file = os.path.join(myDatasetsPath, 'sub_Train', "Train_Dia_Length_Color.txt")
            args.val_json_file = os.path.join(myDatasetsPath, "sub_" + val_set, val_set + "_Dia_Length_Color.txt")

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
    args.base_dir = myDatasetsPath

if args.run_script == 'Inference':
    if args.network_type == 'bbox_detection':
        results_dir = os.path.join(config.General.experiment_path, 'Inf_min_score_'+str(config.Detection.min_score))
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = config.General.experiment_path

    config.DrawProperties.save_img_path = os.path.join(config.General.experiment_path, "Vis_" + val_set)
    os.makedirs(config.DrawProperties.save_img_path, exist_ok=True)

    config.General.files_path = os.path.join(results_dir, "OutputFiles_"+ val_set)
    os.makedirs(config.General.files_path, exist_ok=True)

    args.txt_results = os.path.join(config.General.files_path,
                                    "with_Vis_results_"+val_set +".txt" if config.General.to_draw else
                                    "without_Vis_results_"+val_set +".txt")
    args.output_csv = os.path.join(config.General.files_path,
                                   "with_Vis_results_"+ val_set +".csv" if config.General.to_draw else
                                   "without_Vis_results_"+ val_set +".csv")

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
    if args.network_type == "both":
        args.per_object_weights_dir = os.path.join(args.myExpPath, "Weights", "per_object_counting")
    elif args.network_type == "both_for_roots_2":
        args.per_object_weights_dir = os.path.join(args.myExpPath, "Weights", "per_object_attributes")

    os.makedirs(args.bbox_detection_weights_dir, exist_ok=True)
    os.makedirs(args.per_object_weights_dir, exist_ok=True)
    # "D:\\from 16\\more_counting_Res\\more_counting_Res\\legonet_epoch=249.pt" #cont_legonet_epoch=14.pt"

    if args.load_bbox_det_weights:
        args.bbox_detection_weights_file = get_weights_file(args.bbox_detection_weights_dir)

    if args.load_per_object_counting_weights or args.load_per_object_attributes_weights:
        args.per_object_weights_file = get_weights_file(args.per_object_weights_dir)

    else:
        args.weights_dir = os.path.join(args.myExpPath, "Weights",  "per_object_attributes\\keyPoints_based\\2023-05-23_162315")  # args.network_type, "legonet_epoch=249.pt")
        args.weights_file_path = get_weights_file(args.weights_dir)

        args.model_path = args.weights_file_path # for initial load of old weights file



# else:
#     args.weights_file_path = ""

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