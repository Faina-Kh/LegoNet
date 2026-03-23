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

# Check if GPU is available
config.General.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set GPU if available
if config.General.device.type == 'cuda':
    current_gpu = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))
else:
    print('Running on cpu')



def parse_args(args):
    parser = argparse.ArgumentParser(description='main script.')

    return parser.parse_args(args)


# parse arguments
args = None

if args is None:
    args = sys.argv[1:]
    args = parse_args(args)


########################################################################################################################
# GPU and workers settings
########################################################################################################################

args.num_workers = 0 # 0 - for single processing

args.batch_size = 1

########################################################################################################################
# Define the network options
########################################################################################################################
args.pre_process = 'torch_like' #'keras_like'  # torch_like

args.backbone_type = "ResNetBackboneModule"

#args.dataset_type = "csv_LCC" #"kcsv" #"coco" #'csv_LCC'

args.network_type = "counting_lean"
# detection #"both" #"counting_lean"  # "counting_reg" #"counting_reg"

config.General.NETWORK_TYPE = config.NetworkType.counting_lean
#config.NetworkType.counting_reg  #config.NetworkType.counting_lean
#config.NetworkType.detection_and_counting #config.NetworkType.detection #config.NetworkType.detection_and_counting

assert args.network_type == "detection" or args.network_type == "counting_fat" or args.network_type == "counting_lean" \
       or args.network_type == "counting_reg" or args.network_type == "both"

#assert args.dataset_type == "csv_LCC" or args.dataset_type == "kcsv"



args.output_size = 1

#args.do_Kfold = False


########################################################################################################################
# Define the paths
########################################################################################################################
STORAGE_PATH = r"C:\Users\bordezki\Desktop"
args.dataset_name = "grapes" #"roots"

assert args.dataset_name == "grapes" or args.dataset_name == "roots"

args.dataset_path = paths.get_dataset_path(STORAGE_PATH, args.dataset_name)

if args.dataset_name == 'roots': #args.dataset_type == 'kcsv':
    args.kcsv_train = os.path.join(args.dataset_path, 'train.kcsv')
    args.kcsv_val = os.path.join(args.dataset_path, 'val.kcsv')
    args.kcsv_test = os.path.join(args.dataset_path, 'test.kcsv')
    args.kcsv_classes = os.path.join(args.dataset_path, "classes.kcsv")

elif args.dataset_name == "roots":

    #sample_n = "200"

    val_set = "Test" #"Val" #"Test"
    assert val_set == 'Val' or val_set == 'Test'

    # if args.do_Kfold:
    #     k = "1"
    #     args.train_csv_leaf_number_file = os.path.join(args.dataset_path,'sub_Train_'+k, "Train.csv")
    #     args.train_csv_leaf_location_file = os.path.join(args.dataset_path,'sub_Train_'+k, 'Train_pointsOutput.csv')
    #     args.train_json_file = None
    #
    #     args.val_csv_leaf_number_file = os.path.join(args.dataset_path,"sub_"+val_set +"_"+k,  val_set+".csv" )
    #     args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "sub_"+val_set +"_"+k, val_set+"_pointsOutput.csv")
    #     args.val_json_file = None
    #else:
    args.train_csv_leaf_number_file = os.path.join(args.dataset_path, 'sub_Train', "Train.csv")
    args.train_csv_leaf_location_file = os.path.join(args.dataset_path, 'sub_Train','Train_pointsOutput.csv')
    args.train_json_file = None

    args.val_csv_leaf_number_file = os.path.join(args.dataset_path, "sub_" + val_set, val_set + ".csv")
    args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "sub_" + val_set, val_set + "_pointsOutput.csv")
    args.val_json_file = None

    args.loss_weight = 1 #1  #1000 #10 #100

    config.General.binary_version = ""

############################################################################

args.have_GT = False

if args.have_GT:
    args.base_dir = None  # args.dataset_path
else:
    args.base_dir = args.dataset_path

############################################################################



config.General.dataset_name= args.dataset_name