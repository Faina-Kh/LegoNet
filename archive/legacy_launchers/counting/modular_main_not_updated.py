import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))



###########################################################
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###########################################################

import argparse
import collections
import numpy as np
import config

import torch
# torch.set_default_dtype(torch.double)
import torch.optim as optim
from torchvision import transforms

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


# storagePath
import paths
myStoragePath = paths.STORAGE_PATH
myDatasetsPath = paths.DATASETS_PATH
myExpResultsPath = paths.EXP_RESULTS_PATH
myModelsPath = paths.MODELS_PATH


import legonet.runner

def parse_args(args):
    parser = argparse.ArgumentParser(description='training script.')

    return parser.parse_args(args)


# parse arguments
args = None

if args is None:
    args = sys.argv[1:]
    args = parse_args(args)


########################################################################################################################
# GPU and workers settings
########################################################################################################################

# args.current_gpu = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.current_gpu)
# print('Running on gpu {}'.format(args.current_gpu))

args.num_workers = 0 # 0 - for single processing, was 3

args.batch_size = 1



########################################################################################################################
# Define the network options
########################################################################################################################
args.pre_process = 'torch_like' #'keras_like'  # torch_like

args.backbone_type = "ResNetBackboneModule" #EfficientNetBackboneModule

args.dataset_type = 'kcsv' #"kcsv" #"coco" #'csv_LCC'

args.network_type = "both" #"both" #"counting_lean"  # detection # counting_fat  #counting_lean
config.General.NETWORK_TYPE = config.NetworkType.detection_and_counting #config.NetworkType.detection #config.NetworkType.detection_and_counting

assert args.network_type == "detection" or args.network_type == "counting_fat" or args.network_type == "counting_lean" or args.network_type == "both"
assert args.dataset_type == 'coco' or args.dataset_type == "csv_LCC" or args.dataset_type == "kcsv"


# for counting:
ds = 'A4'
args.lean_version = ""  #"version_2", "version_3" #"version_1"


########################################################################################################################
# Define the paths
########################################################################################################################

if args.dataset_type == "coco":
    args.dataset_name = 'tomato_fruit_12_3_18'

    args.dataset_path = os.path.join(myDatasetsPath, 'Phenomics data', args.dataset_name)

# TODO: THIS IS MY BEST EVER DATA FORMAT FOR EVERYTHING!!!!!!!!!!!!!
elif args.dataset_type == 'kcsv':

    args.dataset_name = "131_wheat_spikes_and_spikelets" #"banana_bunch_segmentation_alt_split" #"131_wheat_spikes_and_spikelets" # "task_122_banana_bunch_detection" #"banana_bunch_segmentation"
    #args.dataset_path = os.path.join(myDatasetsPath,'KK_datasets', args.dataset_name) #'Faina_datasets'
    args.dataset_path = os.path.join(myDatasetsPath,'KK_datasets', args.dataset_name)
    args.kcsv_train = os.path.join(args.dataset_path, "train", 'train_MS5.kcsv') #"train_9K_0.3.kcsv") #"train_9K.kcsv")
    #args.kcsv_val = None
    args.kcsv_val = os.path.join(args.dataset_path, "val", 'val_MS5.kcsv') #"val_9K_0.3.kcsv") #"val_9K.kcsv")
    args.kcsv_test = os.path.join(args.dataset_path, "test", 'test_MS5.kcsv') #"test_9K_0.3.kcsv") #"test_9K.kcsv")
    args.kcsv_classes = os.path.join(args.dataset_path, "classes.kcsv")


elif args.dataset_type == "csv_LCC":

    args.dataset_name = 'Counting Datasets\\CVPPP2017_LCC_training\\training\\' + ds
    args.train_csv_leaf_number_file = os.path.join(myDatasetsPath, args.dataset_name, 'train', ds + '_Train.csv')
    args.train_csv_leaf_location_file = os.path.join(myDatasetsPath, args.dataset_name, 'train',
                                                     ds + '_Train_leaf_location.csv')
    args.val_csv_leaf_number_file = os.path.join(myDatasetsPath, args.dataset_name, 'val', ds + '_Val.csv')
    args.val_csv_leaf_location_file = os.path.join(myDatasetsPath, args.dataset_name, 'val',
                                                   ds + '_Val_leaf_location.csv')




########################################################################################################################
# Train settings
########################################################################################################################

args.epochs = 300



########################################################################################################################
# Choose what script to run
########################################################################################################################

args.run_script = 'train' #'train' #validation
assert args.run_script=='train' or args.run_script=='validation'

args.val_file = args.kcsv_val


########################################################################################################################
# Run the code
########################################################################################################################
config.Detection.BBOX_ADJUSTMENT_RATIO = 1 #1.25

config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False
config.Detection.BBOX_SCALING_FOR_COUNTING_AUGMENTATION = np.linspace(start=0.75, stop=1.25, num=3)



#########################################################################################################################
if args.dataset_name == "131_wheat_spikes_and_spikelets" :
    config.AttributeEstimation.crops_size = [640, 640] #[800, 800]- wheat, [1024,1024]-bananas #[640,640]

    # wheat - (5,5), (5,5), (5,3), (3,5), (3,3)
    # bananas - (5,7), (3, 5), (3, 7), (5, 5), (3,3)
    config.AttributeEstimation.map_1_R = (6, 6) #(5, 5)
    config.AttributeEstimation.map_2_R = (5, 5)
    config.AttributeEstimation.map_3_R = (5, 3)
    config.AttributeEstimation.map_4_R = (3, 5)
    config.AttributeEstimation.map_5_R = (3, 3)

elif args.dataset_name == "banana_bunch_segmentation_alt_split":
    config.AttributeEstimation.crops_size = [1024, 1024]

    config.AttributeEstimation.map_1_R = (5, 7) #(7,7) #(5, 7)
    config.AttributeEstimation.map_2_R = (5, 5) #(6,6) #(5, 5)
    config.AttributeEstimation.map_3_R = (3, 7) #(5,5) #(3, 7)
    config.AttributeEstimation.map_4_R = (3, 5) #(4,4) #(3, 5)
    config.AttributeEstimation.map_5_R = (3, 3)

assert args.dataset_name == "banana_bunch_segmentation_alt_split" or args.dataset_name == "131_wheat_spikes_and_spikelets"

########################################################################################################################
# Run the code
########################################################################################################################
current_dir = "D:\\PyCharmProjects\\Lego2_experiments" #os.path.join("C:\\Users\\khoro\\Desktop","Lego2_experiments",'wheat') #"D:\\PyCharmProjects\\Lego2_experiments"
results_dir = 'try_stuff'#'Ablations\\count_regressor_P3_P5' #'Ablations\\no_inter_losses' #'Ablations\\count_regressor' #'wheatnewannots' #'wheat_9K'  #'bananas_alt_split'

# 'C:\\Users\\Aragorn\\Desktop\\Experiments'
config.General.experiment_path = os.path.join(current_dir, results_dir)#os.path.join(results_dir, )  #"wheat_MS5_no_interloss") #, 'bestsofar', 'min_score_0.7_crop_640')#'9k_0.3')#'320_frozen_detection') #'800_cont_detection') # '320_frozen_detection' #'320_cont_detection' #'models_start 15_other' #'models_start 15_7_6_5_4_3'
if not config.General.experiment_path == '' :
    if not os.path.exists(config.General.experiment_path):
        os.makedirs(config.General.experiment_path)


# True if loading the model (with pre-trained weights)banana_bunch_segmentation
args.use_checkpoint = False

# True- if loading only the pre-trained weights
args.load_weights = False
args.model_path = os.path.join(config.General.experiment_path, 'legonet_final.pt') #230

args.load_partial_weights = False
#args.weights_dir = os.path.join(config.General.experiment_path, 'weights')  #"C:\\Users\\khoro\\Desktop" #"C:\\Users\\Aragorn\\Desktop\\Experiments"
args.partial_weights_dir = os.path.join(current_dir,"weights_detection") #"weights_detection" #"wheat_MS5_s0.9_640\\weights_detection"

args.freeze_detection = False

args.eval_in_train = False
args.train_in_turns = False
config.AttributeEstimation.do_nmcs = True
config.General.to_draw = True

config.Detection.min_score = 0.9 #0.05 #0.95 #0.5 #0.05
config.Detection.iou_threshold = 0.3 #0.5 #[0.3, 0.5, 0.7, 0.9, 0.95] #0.5

config.Detection.min_score_list = [0.05, 0.5, 0.7,0.9]
config.Detection.iou_threshold_list = [0.3, 0.5, 0.7]
config.Detection.NMS_THRESHOLD = 0.3

config.Detect_and_Estimate.choose_by_IoUandPrc_Flag = False
config.Detect_and_Estimate.precision_thresh = 0.3

# currently don't have it
config.Detect_and_Estimate.do_gt_nmcs = False

config.AttributeEstimation.calc_det_performance = True

config.AttributeEstimation.estimate_type = 'reg_fpn_p3_p7_min_sig' #'withKeyPoints' #'reg_fpn_p3_p7_min_sig'
assert config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig' or config.AttributeEstimation.estimate_type == 'withKeyPoints'
config.AttributeEstimation.num_of_pyr_levels = 3

config.AttributeEstimation.inter_losses = True # relevant only to 'withKeyPoints'
########################################################################################################################

legonet.runner.run(args)
#legonet.runner.run_offline_validation(args)


########################################################################################################################
