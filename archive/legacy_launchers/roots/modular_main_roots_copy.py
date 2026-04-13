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

current_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

import warnings
warnings.filterwarnings("ignore")


###########################################################
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###########################################################
from datetime import datetime

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

args.dataset_type = "csv_LCC" #"kcsv" #"coco" #'csv_LCC'

args.network_type = "counting_lean"
#"both" #"counting_lean"  # detection # counting_fat  #"counting_reg"
config.General.NETWORK_TYPE = config.NetworkType.counting_lean
#config.NetworkType.detection_and_counting #config.NetworkType.detection #config.NetworkType.detection_and_counting

assert args.network_type == "detection" or args.network_type == "counting_fat" or args.network_type == "counting_lean" \
       or args.network_type == "counting_reg" or args.network_type == "both"

assert args.dataset_type == 'coco' or args.dataset_type == "csv_LCC" or args.dataset_type == "kcsv"


# for counting:
ds = 'A4'
args.lean_version = ""  #"version_2", "version_3" #"version_1"

args.output_size = 1
args.version = ""


########################################################################################################################
# Define the paths
########################################################################################################################

if args.dataset_type == "coco":
    args.dataset_name = 'tomato_fruit_12_3_18'

    args.dataset_path = os.path.join(myDatasetsPath, 'Phenomics data', args.dataset_name)

# TODO: THIS IS MY BEST EVER DATA FORMAT FOR EVERYTHING!!!!!!!!!!!!!
elif args.dataset_type == 'kcsv':

    args.dataset_name = "grapes" #"131_wheat_spikes_and_spikelets" #"banana_last"

    config.General.dataset_name= args.dataset_name

    args.dataset_path = os.path.join(myDatasetsPath,'KK_datasets', args.dataset_name) #'Faina_datasets'

    args.kcsv_train = os.path.join(args.dataset_path, 'train.kcsv')  #os.path.join(args.dataset_path, "train", 'train_MS5.kcsv') #"train_9K_0.3.kcsv") #"train_9K.kcsv")
    #args.kcsv_val = None
    args.kcsv_val = os.path.join(args.dataset_path, 'val.kcsv') #"val", 'val_MS5.kcsv') #"val_9K_0.3.kcsv") #"val_9K.kcsv")
    args.kcsv_test = os.path.join(args.dataset_path, 'test.kcsv') #"test", 'test_MS5.kcsv') #"test_9K_0.3.kcsv") #"test_9K.kcsv")
    args.kcsv_classes = os.path.join(args.dataset_path, "classes.kcsv")

    assert args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "banana_last" \
           or args.dataset_name == "grapes" or args.dataset_name == "roots"


elif args.dataset_type == "csv_LCC":

    args.dataset_name = "roots"  # "grapes" #"131_wheat_spikes_and_spikelets" #"banana_last"

    if args.dataset_name == "roots":

        dataset = ""  #"original" #"Rootfly_cam3" #"all_manual_2" #"Corn 2020" #"all_manual"
                  #"splitted_data-melon 2018" #"splitted_data-corn 2020" #"Tomato 2020" #"splitted_data-tomato 2019" #"splitted_data-pepper 2021" ##"splitted_data-melon 2018" , # "splitted_data-corn 2020", #"splitted_data- tomato 2020",
        args.dataset_path = os.path.join("D:\\Faina\\roots_project", "Rootfly_cam3\\Aug 22")
                                         #dataset, "Aug 22")
                                         #os.path.join("D:\\Faina\\roots_project", "Autocam image for CNN model", "Images for training\\Training dataset_automated camera", dataset)
                                         #("D:\\Faina\\roots_project", "manual_camera\\July_22", dataset)
                                         #"Autocam image for CNN model", "Images for training\\Training dataset_automated camera", dataset)
                                         #"manual_camera", "13_5_22\\Correction after visual inspection_20220513", dataset)
                                         #"manual_camera\\5_5_22\\Corrected annotation\\pepper_2021",
                                         #"pepperm2021-splitted_data\\all_Cropped")#"joined_data_partial_corrections")

        args.train_csv_leaf_number_file = "" #os.path.join(args.dataset_path, 'sub_Train', 'Train.csv') #'Train_June_try.csv')
        args.train_csv_leaf_location_file = "" #os.path.join(args.dataset_path, 'sub_Train', "Train_pointsOutput.csv") #"Train_pointsOutput_June_try.csv")
        args.train_json_file = None

        args.val_csv_leaf_number_file = os.path.join(args.dataset_path, "raw_images", "autoCam3_test_TRL_raw.csv") #"raw_images", "autoCam3_test_TRL_raw.csv")    #'sub_Val', "Val.csv") #"Test_June_try.csv")  #"pepper_corrected_all.csv") #'sub_Test', "Test.csv") #'sub_Train', "Train.csv")
        args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "raw_images", "pointsOutput.csv") #"raw_images" 'sub_Val', "Val_pointsOutput.csv") #"Test_pointsOutput_June_try.csv") #"pointsOutput_all.csv") #'sub_Test', "Test_pointsOutput.csv") #Test_pointsOutput_2.csv
        args.val_json_file = None

        test_epoch = "198" #"255" #"156" #"242"
        Results_path = os.path.join(args.dataset_path, "Results", "TRL_train manual_test auto_832") #"TRL_train auto_test manual"

        args.txt_results = os.path.join(args.dataset_path, Results_path, "test_epoch"+test_epoch+".txt")
        args.visualize_im = True
        args.save_img_path = os.path.join(args.dataset_path, Results_path, "test_"+test_epoch+"_pred") #_img_0.7")
        if args.save_img_path != "":
            os.makedirs(args.save_img_path, exist_ok=True)
        args.save_detection_eval_path = os.path.join(args.dataset_path, Results_path, "test_"+test_epoch+"_detectEval")
        if args.save_detection_eval_path != "":
            os.makedirs(args.save_detection_eval_path, exist_ok=True)

    else:
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

args.run_script = 'validation' #'train' #validation
assert args.run_script =='train' or args.run_script =='validation'

if args.dataset_type == 'kcsv':
    args.val_file = args.kcsv_val

    if args.val_file == args.kcsv_test:
        config.General.MODE='test'

########################################################################################################################
# Run the code
########################################################################################################################
config.Detection.BBOX_ADJUSTMENT_RATIO = 1 #1.25

config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False
config.Detection.BBOX_SCALING_FOR_COUNTING_AUGMENTATION = np.linspace(start=0.75, stop=1.25, num=3)

########################################################################################################################

# points detection config
if args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "grapes" or args.dataset_name == "roots":
    config.AttributeEstimation.crops_size = [640, 640] #[800, 800]- wheat, [1024,1024]-bananas #[640,640]

    # wheat - (5,5), (5,5), (5,3), (3,5), (3,3)
    # bananas - (5,7), (3, 5), (3, 7), (5, 5), (3,3)
    config.AttributeEstimation.map_1_R = (6, 6) #(5, 5)
    config.AttributeEstimation.map_2_R = (5, 5)
    config.AttributeEstimation.map_3_R = (5, 3)
    config.AttributeEstimation.map_4_R = (3, 5)
    config.AttributeEstimation.map_5_R = (3, 3)

    # for ablation
    # config.Counting.map_1_R = (3, 3)
    # config.Counting.map_2_R = (3, 3)
    # config.Counting.map_3_R = (3, 3)
    # config.Counting.map_4_R = (3, 3)
    # config.Counting.map_5_R = (3, 3)

elif args.dataset_name == "banana_last":
    config.AttributeEstimation.crops_size = [640, 640]

    # config.Counting.map_1_R = (5,7) #(7,7) #(5, 7)
    # config.Counting.map_2_R = (5,5) #(6,6) #(5, 5)
    # config.Counting.map_3_R = (3,7) #(5,5) #(3, 7)
    # config.Counting.map_4_R = (3,5) #(4,4) #(3, 5)
    # config.Counting.map_5_R = (3,3)

    config.AttributeEstimation.map_1_R = (3, 3)  # (5, 5)
    config.AttributeEstimation.map_2_R = (3, 3)
    config.AttributeEstimation.map_3_R = (3, 3)
    config.AttributeEstimation.map_4_R = (3, 3)
    config.AttributeEstimation.map_5_R = (3, 3)




########################################################################################################################
# Run the code
########################################################################################################################

time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

results_dir = "" #os.path.join(Results_path, time_stemp)
#os.path.join("D:\\Faina\\roots_project", "manual_camera", "13_5_22\\Correction after visual inspection_20220513",
                         #  "all_manual" ,'Results')
                           # #(myExpResultsPath, 'KK_Exp_Results_last', time_stemp) #C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults
current_results_dir = "" #"points_for_roots_eval" #"points_for_roots_eval" #"reg_for_roots_eval"


config.General.experiment_path = os.path.join("D:\\Faina\\roots_project", "manual_camera\\July_22\\all_manual_2\\Results\\TRL only\\2022-08-03_160721")
                                              #"Autocam image for CNN model\\Images for training\\Training dataset_automated camera","original\\Results\\TRL only\\2022-07-27_121118")
#os.path.join(results_dir, current_results_dir)
if not config.General.experiment_path == '':
    if not os.path.exists(config.General.experiment_path):
        os.makedirs(config.General.experiment_path)


# True if loading the model (with pre-trained weights)banana_bunch_segmentation
args.use_checkpoint = False

# True- if loading only the trained weights (but not the model)
args.load_weights = True
args.model_path = os.path.join(config.General.experiment_path, "legonet_epoch="+test_epoch+".pt") #'legonet_epoch=89.pt')

args.load_partial_weights_only = False
args.partial_weights_dir = ""
#"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults\\KK_Exp_Results\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
# #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')

args.load_model_and_partial = False

args.freeze_detection = True

args.eval_in_train = True
args.train_in_turns = False
config.AttributeEstimation.do_nmcs = True

#config.General.to_draw = False

config.AttributeEstimation.calc_det_performance = True
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

config.AttributeEstimation.inter_losses = True

config.AttributeEstimation.estimate_type = 'withKeyPoints'

assert config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig' or config.AttributeEstimation.estimate_type == 'withKeyPoints'

config.AttributeEstimation.num_of_pyr_levels = 5 #3

config.Detect_and_Estimate.two_backbones = True
config.Detect_and_Estimate.single_backbone = False
config.Detect_and_Estimate.crop_from_Pi = False

config.Detect_and_Estimate.cancel_nms_in_train = False

########################################################################################################################
# Run training or validation of a specific model
legonet.runner.run(args)

# run validation on multiple models with different min_score and iou_threshold
#legonet.runner.run_offline_validation(args)

#legonet.runner.visualize_detection(args)

########################################################################################################################

