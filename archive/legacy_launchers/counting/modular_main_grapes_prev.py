import os
import sys
import torch
import argparse
import collections
import numpy as np

import config
import paths
import legonet.runner


# torch.set_default_dtype(torch.double)
import torch.optim as optim
from torchvision import transforms
#assert torch.__version__.split('.')[0] == '1'

import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)


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

STORAGE_PATH = r"C:\Users\bordezki\Desktop\LegoNet"
args.dataset_name = "grapes" #"roots"

assert args.dataset_name == "grapes" or args.dataset_name == "roots"

myPaths = paths.get_paths(STORAGE_PATH, args.dataset_name)
myDatasetsPath = myPaths["DATASETS_PATH"]


if args.dataset_type == "coco":
    args.dataset_name = 'tomato_fruit_12_3_18'

    #args.dataset_path = os.path.join(myDatasetsPath, 'Phenomics data', args.dataset_name)


elif args.dataset_type == 'kcsv':

    #args.dataset_name = "grapes" #"131_wheat_spikes_and_spikelets" #"banana_last"

    config.General.dataset_name= args.dataset_name

    #args.dataset_path = os.path.join(myDatasetsPath,'KK_datasets', args.dataset_name) #'Faina_datasets'

    args.kcsv_train = os.path.join(myDatasetsPath, 'train.kcsv')  #os.path.join(args.dataset_path, "train", 'train_MS5.kcsv') #"train_9K_0.3.kcsv") #"train_9K.kcsv")
    #args.kcsv_val = None
    args.kcsv_val = os.path.join(myDatasetsPath, 'val.kcsv') #"val", 'val_MS5.kcsv') #"val_9K_0.3.kcsv") #"val_9K.kcsv")
    args.kcsv_test = os.path.join(myDatasetsPath, 'test.kcsv') #"test", 'test_MS5.kcsv') #"test_9K_0.3.kcsv") #"test_9K.kcsv")
    args.kcsv_classes = os.path.join(myDatasetsPath, "classes.kcsv")

    #assert args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "banana_last" or args.dataset_name == "grapes"

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

# args.val_file = args.kcsv_test
#
# if args.val_file == args.kcsv_test:
#     config.General.MODE='test'

########################################################################################################################
# Run the code
########################################################################################################################
config.Detection.BBOX_ADJUSTMENT_RATIO = 1 #1.25

config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False
config.Detection.BBOX_SCALING_FOR_COUNTING_AUGMENTATION = np.linspace(start=0.75, stop=1.25, num=3)

#########################################################################################################################
if args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "grapes":
    config.Counting.crops_size = [640,640] #[800, 800]- wheat, [1024,1024]-bananas #[640,640]

    # wheat - (5,5), (5,5), (5,3), (3,5), (3,3)
    # bananas - (5,7), (3, 5), (3, 7), (5, 5), (3,3)
    config.Counting.map_1_R = (6,6) #(5, 5)
    config.Counting.map_2_R = (5,5)
    config.Counting.map_3_R = (5,3)
    config.Counting.map_4_R = (3,5)
    config.Counting.map_5_R = (3,3)

    # for ablation
    # config.Counting.map_1_R = (3, 3)
    # config.Counting.map_2_R = (3, 3)
    # config.Counting.map_3_R = (3, 3)
    # config.Counting.map_4_R = (3, 3)
    # config.Counting.map_5_R = (3, 3)

elif args.dataset_name == "banana_last":
    config.Counting.crops_size = [640, 640]

    # config.Counting.map_1_R = (5,7) #(7,7) #(5, 7)
    # config.Counting.map_2_R = (5,5) #(6,6) #(5, 5)
    # config.Counting.map_3_R = (3,7) #(5,5) #(3, 7)
    # config.Counting.map_4_R = (3,5) #(4,4) #(3, 5)
    # config.Counting.map_5_R = (3,3)

    config.Counting.map_1_R = (3, 3)  # (5, 5)
    config.Counting.map_2_R = (3, 3)
    config.Counting.map_3_R = (3, 3)
    config.Counting.map_4_R = (3, 3)
    config.Counting.map_5_R = (3, 3)




########################################################################################################################
# Run the code
########################################################################################################################

results_dir = myPaths["EXP_RESULTS_PATH"] #'C:\\Users\\Aragorn\\Desktop'  #os.path.join('C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults', 'KK_Exp_Results_last') #C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults #myExpResultsPath
current_results_dir = "twoBack_keyP_diff radii"
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


config.General.experiment_path = os.path.join(results_dir, current_results_dir)
if not config.General.experiment_path == '':
    if not os.path.exists(config.General.experiment_path):
        os.makedirs(config.General.experiment_path)


# True if loading the model (with pre-trained weights)banana_bunch_segmentation
args.use_checkpoint = False

# True- if loading only the pre-trained weights
args.load_weights = True
args.model_path = os.path.join("C:\\Users\\Aragorn\\Desktop",'legonet_epoch=249.pt')
#'C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults (1)\\KK_Exp_Results_last\\grapes_twoBack_keyP_sameRadii_perfect det_fancy aug_20Per\\legonet_epoch=154.pt'
#os.path.join(config.General.experiment_path,'legonet_epoch=264.pt')
#os.path.join(config.General.experiment_path,'legonet_epoch=214.pt')
#os.path.join(results_dir, "grapes_twoBack_keyP_different radii_fixed",'legonet_epoch=249.pt')
#os.path.join(config.General.experiment_path, 'legonet_epoch=249.pt') #'legonet_epoch=229.pt')
#os.path.join(config.General.experiment_path, 'legonet_epoch=194.pt')
#os.path.join(config.General.experiment_path, 'legonet_epoch=154.pt')
#os.path.join(results_dir,"grapes_twoBack_reg_P3_P5_fluid_new", 'legonet_epoch=164.pt')

args.load_partial_weights_only = False

args.partial_weights_dir = "" #"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults (1)\\KK_Exp_Results_last\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
# #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')
#os.path.join(results_dir,'grapes_twoBack_reg_P3_P5_fluid_new', 'saved_weights_164')

args.load_model_and_partial = False

args.freeze_detection = False

args.eval_in_train = True
args.train_in_turns = False
config.Counting.do_nmcs = True

config.General.to_draw = True
config.Counting.calc_det_performance = True
config.DrawProperties.DRAW_MAPS= True

config.Detection.min_score = 0.7 #0.95 #0.5 #0.05
config.Detection.iou_threshold = 0.5 #0.5 #[0.3, 0.5, 0.7, 0.9, 0.95] #0.5

config.Detection.min_score_list = [0.7]
config.Detection.iou_threshold_list = [0.3, 0.5, 0.7]
config.Detection.NMS_THRESHOLD = 0.3

config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False

config.Detection.USE_PERFECT_DETECTION_MODE = False

#config.detect_and_count.choose_by_IoUandPrc_Flag = False
#config.detect_and_count.precision_thresh = 0.3
#config.detect_and_count.do_gt_nmcs = False

config.Counting.inter_losses = True

config.Counting.counting_type = 'withKeyPoints'

assert config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig' or config.Counting.counting_type == 'withKeyPoints'

config.Counting.num_of_pyr_levels = 5

config.detect_and_count.two_backbones = True
config.detect_and_count.single_backbone = False
config.detect_and_count.crop_from_Pi = False


config.detect_and_count.cancel_nms_in_train = False

########################################################################################################################
# Run training or validation of a specific model
legonet.runner.run(args)

# run validation on multiple models with different min_score and iou_threshold
#legonet.runner.run_offline_validation(args)

#legonet.runner.visualize_detection(args)

########################################################################################################################

