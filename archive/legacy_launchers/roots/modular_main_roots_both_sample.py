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

import time
startTime = time.time()

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

#assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

config.General.current_gpu = int(current_gpu)

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

args.dataset_type = "roots_json" #"csv_LCC" #"kcsv" #"coco" #'csv_LCC'

args.network_type = "both_for_roots_2"  #"detection"
#"both" #"both_for_roots" #"counting_lean"  # detection # counting_fat  #"counting_reg"

config.General.NETWORK_TYPE = config.NetworkType.detection_and_counting
#config.NetworkType.detection #config.NetworkType.detection_and_counting
config.detect_and_count.type = args.network_type

assert args.network_type == "detection" or args.network_type == "counting_fat" or args.network_type == "counting_lean" \
       or args.network_type == "counting_reg" or args.network_type == "both" or args.network_type == "both_for_roots" or args.network_type == "both_for_roots_2"

assert args.dataset_type == 'coco' or args.dataset_type == "csv_LCC" or args.dataset_type == "kcsv" or args.dataset_type=="roots_json"


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

        #dataset = "splitted_data-melon 2018" #"splitted_data-corn 2020" #"Tomato 2020" #"splitted_data-tomato 2019" #"splitted_data-pepper 2021" ##"splitted_data-melon 2018" , # "splitted_data-corn 2020", #"splitted_data- tomato 2020",
        args.dataset_path = os.path.join("D:\\Faina\\roots_project", "manual_camera", "corrected", "Tomato 2020")
                                          # "manual_camera", "No_corrections", "Tomato 2020")
                                            #"Rootfly_cam3")

        args.train_csv_leaf_number_file = os.path.join(args.dataset_path, 'sub_Train','Train.csv')
        args.train_csv_leaf_location_file = os.path.join(args.dataset_path, 'sub_Train', "Train_pointsOutput.csv")

        args.val_csv_leaf_number_file = os.path.join(args.dataset_path, 'sub_Test', "Test.csv") #'sub_Train', "Train.csv")
        args.val_csv_leaf_location_file = os.path.join(args.dataset_path,'sub_Test', "Test_pointsOutput.csv")

        args.txt_results = "" #os.path.join(args.dataset_path, "Results", "test_Rootfly_cam3_epoch38_points.txt") #"test_epoch78_points_Results.txt")

        args.visualize_im = False
        args.save_img_path = "" #os.path.join(args.dataset_path, "Results", "test_Rootfly_cam3_epoch38_img")
        if args.save_img_path != "":
            os.makedirs(args.save_img_path, exist_ok=True)
        args.save_detection_eval_path = os.path.join(args.dataset_path, "Results", "test__Rootfly_cam3_epoch38_points")
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

elif args.dataset_type=="roots_json":
    args.dataset_name = "roots"

    config.General.dataset_type = "roots_json"

    #############################################################################
    config.detect_with_points.detect_points = False #old version, didn't work

    config.Detection.change_anchors = True
    config.Detection.ratios = np.array([0.5, 1, 3]) #np.array([0.5, 1, 3]) #np.array([0.5, 1, 4]) #np.array([0.5, 1, 2])

    print("Detection.ratios:", config.Detection.ratios)
    #############################################################################

    #k = "4"
    sample_n = "200"
    args.dataset_path = os.path.join("E:\\roots_project", "Diff_sample")
                                    #"Grapes_K_fold", "K_" + k)
                                    # , "Tube 20_renamed")
                                     #"Demonstration of color and diameter change") #"Joined datasets")
                                     #"Three_datasets_detection")
                                     #"Grapevine_data_all")
                                     #"newAutoCam_17_7_22\\original_28_7_22")
                                     #"Grapevine_data_all")
                                     #"Rootfly_subfolders_Grapevine roots")
                                     #"Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\original")

    Val_and_Test_path = os.path.join("E:\\roots_project", "17_1_data\\For root color model\\all images")

    val_set = "Test"

    args.train_csv_leaf_number_file = os.path.join(args.dataset_path, 'sub_Train_'+sample_n, 'Train_'+sample_n+'_new.csv')   #'Train.csv' ) #'Train_limit_5.csv')
    args.train_csv_leaf_location_file = os.path.join(args.dataset_path, 'sub_Train_'+sample_n, 'Train_pointsOutput_'+sample_n+'_new.csv') #"Train_pointsOutput.csv") #"Train_pointsOutput_limit_5.csv")
    args.train_json_file = os.path.join(args.dataset_path, 'sub_Train_'+sample_n, "Train_Dia_Length_Color_" + sample_n + "_new.txt") #"Train_Dia_Length_Color.txt") #"Train_Dia_and_Length.txt" "Train_all_info.txt" "Train_Dia_Length_Color_limit_5.txt"

    args.val_csv_leaf_number_file = os.path.join(Val_and_Test_path,  "sub_"+val_set,  val_set+".csv") #"Val.csv")  # "Val_limit_5.csv"
    args.val_csv_leaf_location_file = os.path.join(Val_and_Test_path, "sub_"+val_set, val_set+"_pointsOutput.csv") # "Val_pointsOutput.csv") #"Val_pointsOutput_limit_5.csv")  #'sub_Val', "Val_pointsOutput.csv")
    args.val_json_file = os.path.join(Val_and_Test_path,  "sub_"+val_set, val_set+"_Dia_Length_Color.txt") # "Val_Dia_Length_Color.txt") #"Val_all_info.txt") # "Val_Dia_Length_Color_limit_5.txt")  # "Val_Dia_Length_Color.txt"  #"Val_Dia_and_Length.txt"

    args.base_dir = None  # args.dataset_path

    test_epoch = "" #"111" #"119" #"196" #"229" #"90" #"128" #"229" #"217" # "290" #"90" #"80" #"69" # "2" #"231" #"198" #"15" #"231" #"85" #"231"
    weights_dir = "" #"2024-06-19_194041" #"2024-06-18_152231" #"2024-06-18_123358" # "2023-05-28_194055" #"2023-05-23_162315" #"2023-06-27_025106" # "2023-05-28_194055" #"2023-05-24_155926" # "2023-05-25_175752" #"2023-05-23_162315" #"2023-05-23_162613" #"2023-05-20_230659" # "2023-05-18_183327" #"2023-02-28_184758" #"2023-03-05_111030" #"2023-03-04_102708" #"2023-03-03_232404" #"2023-02-28_184758" #"2023-02-28_184758" #"2023-02-28_125804"

    config.Detection.iou_threshold = 0.5 #0.7 #0.9  # 0.5 #[0.3, 0.5, 0.7, 0.9, 0.95] #0.5
    config.Detection.min_score = 0.7 # 0.8  #0.7 #0.95 #0.5 #0.05

    args.dia_loss_weight = 10 #100  # dia weight # 10 #1000 #10 #100
    args.color_loss_weight = 100  # 100
    args.maps_loss_weight = 1 #10

    config.prev_color_model = False

    config.General.predict_empty_image = True

    #################################################################
    args.have_GT = True
    #################################################################


    #################################################################
    config.detect_and_count.use_new_Find = False
    #################################################################

    config.General.with_new_layers_for_both = True  # always True??

    # "both_detect by points" # "both" #"detection" #"both_2_with detect weights", "both_2_detect_3Sets","reg", "eval_mixed_weights" #"both_2_with detect",
    args.Results_path = os.path.join(args.dataset_path, "Results_"+sample_n, "both_roots_2_Reg") #  "both_roots_2_Reg"#"both_roots_2" #, "both_2_detect_3Sets\\reg", "I0.5_s0.7_10_dia_100_color\\trained_all")
                                    # "eval_mixed_weights") # "trained_all" #"eval_mixed_weights", "check_back2b_find_2b_det3grapes")  #"changed_colorModel\\all_I_0.7_s_0.7")
                                    #"newFind",  "all_" # "limit_5_"# "all_"        #\\changed_colorModel",
                                    #  + "I_" +str(config.Detection.iou_threshold)+ "_s_"+str(config.Detection.min_score))
                                    # "both_2_with detect" , "changed_colorModel",
                                    #  "I_" +str(config.Detection.iou_threshold)+ "_s "+str(config.Detection.min_score)+"_limit_5_1_dia_1_color")
                                    #"detection",
                                    #)  # "both_2_detect_3Sets","reg", #"sameFind",
                                    # "I" + str(config.Detection.iou_threshold) +
                                    # "_s"+ str(config.Detection.min_score)+"_limit_5")
                                    #                 +str(args.dia_loss_weight)+"_dia_" +str(args.color_loss_weight)+"_color",
                                    # "trained_all") #"trained_limit5"
    # "I0.5_s0.7_10_dia_100_color")
    # "I 0.7_s 0.7_limit_5_10_dia and 100_color")
    #"detection") ##"Results","both_2_with detect\\changed_colorModel\\I 0.7_s 0.7_limit_5_10_dia and 100_color\\all_fixed detect")
                                     #"detection", "IOU 0.9_score 0.7_limit_5")
    #"IOU 0.7_score 0.5_limit_5") #"train_IOU 0.5_score 0.7")
    #"both_2_with detect", "changed_colorModel" ,#"changed_findModule", #"detection", #"both_2_with detect weights",
                     #                "I "+str(config.Detection.iou_threshold)+
                     #                "_s "+ str(config.Detection.min_score)
                     #                + "_limit_5_"+str(args.dia_loss_weight)+"_dia and " +str(args.color_loss_weight)+" _color",
                     #                "train all_fixed detect")
                                     #"onlyFind2")
                                     #"_limit_5_dia and color 10loss", "cont_epoch 186") #"_limit_5_10_dia and 100_color")
                                     #"_limit_5")
                                     #+"_limit_5_dia and color 10loss")
                                     #"detection", "IOU "+str(config.Detection.iou_threshold)+ "_score "+str(config.Detection.min_score) + "_limit_5")
    os.makedirs(args.Results_path, exist_ok=True)

    args.test_dir = os.path.join(args.Results_path, val_set+"_comp241")
    #, "Test_I_0.5_s_0.7_3")
                                 #"find2_b_backbone_2_b_I_"+str(config.Detection.iou_threshold)+  "_s_"+str(config.Detection.min_score)+"_"+val_set)
                                # "find2_b_backbone_2_b" + val_set+"_I "+str(config.Detection.iou_threshold)+ "_s "+str(config.Detection.min_score))#+"_limit_5" +"_epoch"+test_epoch)
                                 #"_check ratio_2")
                                 #"ratio_4", val_set+"_I "+str(config.Detection.iou_threshold)+ "_s "+str(config.Detection.min_score)+"_all" +"_epoch"+test_epoch)
                                 #"val_"+test_epoch+ "s_"+str(config.Detection.min_score)) #"val_"+weights_dir+"_epoch_"+test_epoch+
    if args.test_dir != "":
        os.makedirs(args.test_dir, exist_ok=True)

    #args.txt_results = os.path.join(args.test_dir, "results.txt") #"val", "val_epoch"+test_epoch+".txt")

    if args.test_dir != "":
        os.makedirs(args.test_dir, exist_ok=True)
        args.txt_results =os.path.join(args.test_dir, "results.txt") #"val", "val_epoch"+test_epoch+".txt")
    else:
        args.txt_results = os.path.join( args.Results_path, "results_train.txt")

    args.visualize_im = False
    args.save_img_path = "" #os.path.join(args.test_dir, "vis") #"val", "val_"+test_epoch+"_pred") #"Results", "test_Rootfly_cam3_epoch38_img")
    if args.save_img_path != "":
        os.makedirs(args.save_img_path, exist_ok=True)
    args.save_detection_eval_path = "" # os.path.join(args.test_dir , "detection_eval")
    if args.save_detection_eval_path != "":
        os.makedirs(args.save_detection_eval_path, exist_ok=True)

########################################################################################################################
# Train settings
########################################################################################################################

args.epochs = 300 #185 #247 #232 #300

args.separate_training = False

args.evaluate_detection = False

args.eval_detection_params = False

args.evaluate_both = True # relevant to validation, not train

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
if args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "grapes" or \
        args.dataset_name == "roots":
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

time_stemp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

if args.run_script == 'train':
    config.General.experiment_path = os.path.join(args.Results_path, time_stemp) #args.test_dir
else:
    config.General.experiment_path = args.test_dir

if not config.General.experiment_path == '':
    if not os.path.exists(config.General.experiment_path):
        os.makedirs(config.General.experiment_path)

########################################################################
args.do_counting = True
########################################################################

# True if loading the model (with pre-trained weights)banana_bunch_segmentation
args.use_checkpoint = False

args.model_path = os.path.join(args.Results_path, weights_dir, "legonet_epoch="+test_epoch+".pt")
#os.path.join("C:\\Users\\Aragorn\Desktop\\roots project\\Grapevine_data_all", "Results",
#                                "both_2_detect_3Sets\\I0.5_s0.7_10_dia_100_color", "2023-05-23_162315", "legonet_epoch=90.pt")
   # os.path.join("C:\\Users\\Aragorn\Desktop\\roots project\\Grapevine_data_all", "Results",
   #                            "both_2_detect_3Sets\\reg", "I0.5_s0.7_10_dia_100_color\\trained_all", weights_dir, "legonet_epoch="+test_epoch+".pt")
    #args.Results_path,  weights_dir, "legonet_epoch="+test_epoch+".pt")
#os.path.join(args.dataset_path, "Results\\both_2_with detect\\changed_colorModel\\I0.7_s 0.7_limit5_10dia_100_color\\fixed detect",
                                                  #"2023-02-28_184758", "legonet_epoch=231.pt")
                  #args.Results_path, weights_dir, "legonet_epoch="+test_epoch+".pt")
#os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project","Grapevine_data_all",
                            #   "Results", "both_2_with detect\\changed_colorModel",
                            #   "I 0.7_s 0.7_limit_5_10_dia and 100_color\\all_fixed detect\\2023-02-28_184758", "legonet_epoch=231.pt")

#os.path.join(args.Results_path, weights_dir, "legonet_epoch="+test_epoch+".pt")
#"C:\\Users\\Aragorn\\Desktop\\roots project\\Grapevine_data_all\\Results\\both_2_with detect\\changed_colorModel\\"
#                        "IOU 0.7_score 0.7_limit_5_10_dia and 100 _color", "2023-02-28_184758","legonet_epoch=231.pt")
#weights_dir, "legonet_epoch="+test_epoch+".pt")
# "Results","both_2_with detect weights", #"detection", #"both_2_with detect weights",
#      "IOU "+str(config.Detection.iou_threshold)+
#                "_score "+ str(config.Detection.min_score)+ "_limit_5_dia and color 10loss",#args.Results_path,
#          weights_dir, "legonet_epoch="+test_epoch+".pt")
#rgs.dataset_path,, "Results\\both_2\\train_score 0.95_IOU 0.5\\limit_5\\2023-02-14_222052", 'legonet_epoch=77.pt')
#"D:\\Faina\\roots_project\\Rootfly_cam3\\Results\\both_for_roots_eval", 'legonet_epoch=179.pt')


# True- if loading only the pre-trained weights (but not the model)
args.load_weights = True
args.load_partial_weights_only = False # True means only detection weights
args.load_model_and_partial = False # True for composing pre-trained network

args.partial_weights_dir = "" #os.path.join(args.dataset_path, "Results_"+sample_n , "Detection", weights_dir , "saved_weights_epoch_"+test_epoch)
    #("C:\\Users\\Aragorn\\Desktop\\roots project", "Three_datasets_detection","Results", "detection","2023-05-20_230659","saved_weights_epoch_69")
                                       # "IOU 0.7_score 0.7_limit_5\\2023-02-15_154227", "saved_weights_epoch_280")  #  "2023-05-20_230659","saved_weights_epoch_69"
                                    # "Results\\detection\\I_0.7_s_0.7\\2023-06-21_153641\\saved_weights_epoch_189")
                                    # "Three_datasets_detection", "Results", "detection", "2023-05-20_230659","saved_weights_epoch_69")

args.load_more_partial = False
limit5Path = "" #os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
                          #"Results\\both_2_with detect", "changed_colorModel","2023-02-28_184758", "saved_weights_epoch_231")
                                              #should be in (path too long) : "I0.7_s 0.7_limit5_10dia_100color", #"fixed_detect", p
                                              # "2023-02-28_184758", "saved_weights_epoch_231")

all_3setsPath = "" #os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all",
                          #  "Results\\both_2_detect_3Sets\\I0.5_s0.7_10_dia_100_color", "2023-05-23_162315","saved_weights_epoch_90")

args.additional_modules_weights =  ""
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
#"Results\\detection", "2023-02-05_193648_ratio 3_val 0.393", " saved_weights_epoch_175")
#"C:\\Users\\Aragorn\\Google Drive\\StoragePath\\ExpResults\\KK_Exp_Results\\grapes_twoBack_keypoints_sameRadi_fluid_new\\saved_weights_cont_14"
# #os.path.join(results_dir, 'wheat_MS5_s0.7_640\\saved_weights_211\\detector_weights')


args.freeze_detection = True

args.freeze_all_except_find_2 = False

args.eval_in_train = True

args.train_in_turns = False
config.Counting.do_nmcs = False

config.Counting.calc_det_performance = False

config.General.to_draw = args.visualize_im
config.DrawProperties.DRAW_MAPS= args.visualize_im
args.draw_maps = False

config.Detection.min_score_list = [0.05, 0.85, 0.9, 0.95] #[0.7]
config.Detection.iou_threshold_list = [0.3, 0.5, 0.7, 0.9]
config.Detection.NMS_THRESHOLD = 0.3

config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False

config.Detection.USE_PERFECT_DETECTION_MODE = False

#config.detect_and_count.choose_by_IoUandPrc_Flag = False
#config.detect_and_count.precision_thresh = 0.3
#config.detect_and_count.do_gt_nmcs = False

config.Counting.inter_losses = True

config.Counting.counting_type = 'reg_fpn_p3_p7_min_sig'

assert config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig' or config.Counting.counting_type == 'withKeyPoints'

config.Counting.num_of_pyr_levels = 5 #3

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


print()
executionTime = (time.time() - startTime)
print('Execution time in minutes: ' + str(executionTime/60))

if args.txt_results != "":
    with open(args.txt_results, 'a') as f:
        f.write('Execution time in minutes: ' + str(executionTime/60))

########################################################################################################################

