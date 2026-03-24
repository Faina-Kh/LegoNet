import os
import sys
import argparse
import torch
import time
import config
import paths

import warnings
warnings.filterwarnings("ignore")

startTime = time.time()

########################################################################################################################
# GPU settings
########################################################################################################################

# Check if GPU is available
config.General.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set GPU if available
if config.General.device.type == 'cuda':
    current_gpu = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))
else:
    print('Running on cpu')

########################################################################################################################

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
STORAGE_PATH = r"C:\Users\bordezki\Desktop"
args.dataset_name = "grapes" #"roots"
args.network_type = "detection"

args.run_script = 'validation' #'train' #validation
assert args.run_script =='train' or args.run_script =='validation'

args.have_GT = True

if args.run_script =='train':
    assert args.have_GT == True
    args.epochs = 300


args.num_workers = 0 # 0 - for single processing

args.batch_size = 1

args.output_size = 1

args.do_Kfold = False

# ToDo - to resolve
######################################################################
args.pre_process = 'torch_like' #'keras_like'  # torch_like
args.backbone_type = "ResNetBackboneModule"
args.lean_version = ""
args.loss_weight = 1  # 1  #1000 #10 #100
config.General.binary_version = ""
config.General.dataset_name= args.dataset_name

######################################################################

assert args.dataset_name == "grapes" or args.dataset_name == "roots"
assert (args.network_type == "detection" or args.network_type == "counting_lean" or args.network_type == "counting_reg" or
        args.network_type == "both" or args.network_type == "both_for_roots_2")

if args.run_script == 'train':
    val_set = "Val"
    assert val_set == 'Val' or val_set == 'Test'

if args.have_GT:
    args.base_dir = None
else:
    args.base_dir = args.dataset_path

########################################################################################################################
# Define the data format and network options
########################################################################################################################

myPaths = paths.get_paths(STORAGE_PATH, args.dataset_name)
myDatasetsPath = myPaths["DATASETS_PATH"]

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


if args.network_type == "detection":
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

if args.dataset_type == 'kcsv':
    args.kcsv_train = os.path.join(myDatasetsPath, 'train.kcsv')
    args.kcsv_val = os.path.join(myDatasetsPath, 'val.kcsv')
    args.kcsv_test = os.path.join(myDatasetsPath, 'test.kcsv')
    args.kcsv_classes = os.path.join(myDatasetsPath, "classes.kcsv")

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

############################################################################


# task specific definitions - organize per task type

args.separate_training = False
config.prev_color_model = False
args.freeze_all_except_find_2 = False