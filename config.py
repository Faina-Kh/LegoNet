import enum
import numpy as np

class DrawProperties:
    POINT_RADIUS = 5
    LINE_WIDTH = 5
    DRAW_MAPS = True
    save_img_path = ''
    maps_path = ''

class NetworkType(enum.Enum):
    detection = 1
    counting_lean = 2
    counting_fat = 3
    detection_and_counting = 4
    counting_reg = 5
    #points_detection_and_evaluation = 6
    counting_lean_multiple_out = 6
    default = 7

class General:
    MODE = "train"
    NETWORK_TYPE = ""#NetworkType.counting_lean_multiple_out
    SAVE_EVERY_N_EPOCHS = 5
    experiment_path = ""
    to_draw = True
    dataset_name=''
    files_path = ""
    #binary_model = False
    #binary_loss_version = ""
    dataset_type = ""
    current_gpu = ""
    #with_new_layers = True
    twoFind_2 = False
    twoBackbone_2 = False
    predict_empty_image = False
    model_name = ""
    device = ""
    filter_empty_bbox = False
    weights_dir = ""
    partial_weights_dir = ""

class Detection:
    backbone_type = "ResNetBackboneModule"
    NMS_THRESHOLD = 0.5
    BBOX_ADJUSTMENT_RATIO = 1.0
    DO_BBOX_AUGMENTATION_FOR_COUNTING = False
    # BBOX_SCALING_FOR_COUNTING_AUGMENTATION = np.linspace(start=0.5, stop=1.5, num=11)
    iou_threshold = 0.5
    min_score = 0.05 #0.7
    iou_threshold_list = [0.5]
    min_score_list = []
    USE_PERFECT_DETECTION_MODE = False
    change_anchors = False
    ratios=None


class AttributeEstimation:
    crops_size = [640,640]
    map_1_R = (6, 6)
    map_2_R = (5, 5)
    map_3_R = (5, 3)
    map_4_R = (3, 5)
    map_5_R = (3, 3)

    calc_det_performance = True
    do_nmcs = False
    estimate_type = "" #'reg_fpn_p3_p7_min_sig' #('reg_fpn_p3' #) #'withKeyPoints'
    inter_losses = True
    num_of_pyr_levels = 3
    double_counting = False
    Find_for_count = True

class Detect_and_Estimate:
    choose_by_IoUandPrc_Flag = True
    precision_thresh = 0.9
    do_gt_nmcs = True
    single_backbone = False
    cancel_nms_in_train = False
    balance_losses = False
    type = ""
    use_new_Find = False
    crop_from_Pi = False

# class detect_with_points:
#     detect_points = False

class roots_ablations:
    freeze_all_except_find_2 = False
