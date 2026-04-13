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

import torch
import config
from datetime import datetime
import numpy as np
import argparse


import legonet.runner

###########################################################
# Check if GPU is available
config.General.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set GPU if available
if config.General.device.type == 'cuda':
    current_gpu = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))
else:
    print('Running on cpu')

###########################################################
import time
startTime = time.time()
###########################################################

import warnings
warnings.filterwarnings("ignore")
###########################################################

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

args.num_workers = 0 # 0 - for single processing

args.batch_size = 1

########################################################################################################################
# Define the network options
########################################################################################################################
args.pre_process = 'torch_like' #'keras_like'  # torch_like

args.backbone_type = "ResNetBackboneModule" #EfficientNetBackboneModule

args.dataset_type = "csv_LCC" #"kcsv" #"coco" #'csv_LCC'

args.network_type = "counting_lean" # "counting_reg" #"counting_lean"
#"both" #"counting_lean"  # detection # counting_fat  #"counting_reg"
config.General.NETWORK_TYPE = config.NetworkType.counting_lean
#config.NetworkType.counting_reg  #config.NetworkType.counting_lean
#config.NetworkType.detection_and_counting #config.NetworkType.detection #config.NetworkType.detection_and_counting

assert args.network_type == "detection" or args.network_type == "counting_fat" or args.network_type == "counting_lean" \
       or args.network_type == "counting_reg" or args.network_type == "both"

assert args.dataset_type == 'coco' or args.dataset_type == "csv_LCC" or args.dataset_type == "kcsv"


# for counting:
ds = 'A4'
args.lean_version = ""  #"version_2", "version_3" #"version_1"

args.output_size = 1
args.version = ""

args.do_Kfold = False


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



        #sample_n = "200"

        args.dataset_path = os.path.join("D:\\Faina", "", "Sharon", "bi-weekly", "Session 10")

        val_set = "Test" #"Val" #"Test"
        assert val_set == 'Val' or val_set == 'Test'

        if args.do_Kfold:
            k = "1"
            args.train_csv_leaf_number_file = os.path.join(args.dataset_path,'sub_Train_'+k, "Train.csv")
            args.train_csv_leaf_location_file = os.path.join(args.dataset_path,'sub_Train_'+k, 'Train_pointsOutput.csv')
            args.train_json_file = None

            args.val_csv_leaf_number_file = os.path.join(args.dataset_path,"sub_"+val_set +"_"+k,  val_set+".csv" )
            args.val_csv_leaf_location_file = os.path.join(args.dataset_path, "sub_"+val_set +"_"+k, val_set+"_pointsOutput.csv")
            args.val_json_file = None
        else:
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
        # define model type
        config.General.binary_model = False
        config.General.with_new_layers = False  # should be False only for points TRL
        #config.General.other = True
        config.General.with_new_layers_for_both = False # always False here
        ############################################################################

        args.Results_path = os.path.join(args.dataset_path, "Results")
        os.makedirs(args.Results_path, exist_ok=True)

        args.test_dir = os.path.join(args.Results_path)
        if args.test_dir != "":
            os.makedirs(args.test_dir, exist_ok=True)

        args.visualize_im = False
        args.save_img_path = ""
        if args.save_img_path != "":
            os.makedirs(args.save_img_path, exist_ok=True)


        if args.test_dir != "":
            os.makedirs(args.test_dir, exist_ok=True)
            if  args.visualize_im:
                args.txt_results = os.path.join(args.test_dir, "with_Vis_results.txt")
            else:
                args.txt_results = os.path.join(args.test_dir, "without_Vis_results.txt")
        else:
            args.txt_results = os.path.join(args.Results_path, "results_train.txt")


        args.save_detection_eval_path = ""
        if args.save_detection_eval_path != "":
            os.makedirs(args.save_detection_eval_path, exist_ok=True)

        ############################################################################

        test_epoch = "45"
        weights_dir = "2025-07-17_184204"

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

args.epochs = 300 #300

args.separate_training = False
config.prev_color_model = False
args.freeze_all_except_find_2 = False



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

if args.run_script == 'train':
    config.General.experiment_path = os.path.join(args.Results_path, time_stemp) #args.test_dir
else:
    config.General.experiment_path = args.test_dir

if not config.General.experiment_path == '':
    if not os.path.exists(config.General.experiment_path):
        os.makedirs(config.General.experiment_path)


# True if loading the model (with pre-trained weights)banana_bunch_segmentation
args.use_checkpoint = False

# True- if loading only the trained weights (but not the model)
args.load_weights = True
args.model_path =  os.path.join("D:\\Faina", "Roots\\Sharon","Hatzeva_all_images_daily\\annotations" ,
                                "splitted\\Results_after_821 and retraining", weights_dir, "legonet_epoch="+test_epoch+".pt")



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

config.AttributeEstimation.calc_det_performance = False
config.DrawProperties.DRAW_MAPS= False

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

executionTime = (time.time() - startTime)
print('Execution time in minutes: ' + str(executionTime/60))

if args.txt_results != "":
    with open(args.txt_results, 'a') as f:
        f.write('Execution time in minutes: ' + str(executionTime/60))

########################################################################################################################

