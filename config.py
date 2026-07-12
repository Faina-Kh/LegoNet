import enum


class DrawProperties:
    POINT_RADIUS = 5
    LINE_WIDTH = 5
    DRAW_MAPS = False
    save_img_path = ''
    maps_path = ''

class NetworkType(enum.Enum):
    detection = 1
    per_image_estimation_keypoints = 2
    detection_and_counting = 3
    per_image_estimation_regression = 4


    # Deprecated aliases retained for compatibility with existing callers.
    counting_lean = per_image_estimation_keypoints
    counting_reg = per_image_estimation_regression


class General:
    MODE = "train"
    NETWORK_TYPE = ""
    SAVE_EVERY_N_EPOCHS = 5
    experiment_path = ""
    to_draw = False
    dataset_name=''
    files_path = ""
    predict_empty_image = True
    model_name = ""
    device = ""
    filter_empty_bbox = False
    weights_dir = ""


class Detection:
    NMS_THRESHOLD = 0.3 #0.5
    iou_threshold = 0.5
    min_score = 0.7 #0.05 #0.7
    iou_threshold_list = [0.3, 0.5, 0.7]
    min_score_list = []
    change_anchors = False
    ratios=None

class AttributeEstimation:
    crops_size = [640,640]
    map_1_R = (6, 6)
    map_2_R = (5, 5)
    map_3_R = (5, 3)
    map_4_R = (3, 5)
    map_5_R = (3, 3)

    calc_det_performance = False
    do_nmcs = False
    estimate_type = "" #'reg_fpn_p3_p7_min_sig' #('reg_fpn_p3' #) #'withKeyPoints'
    inter_losses = True
    num_of_pyr_levels = 5

class Detect_and_Estimate:
    precision_thresh = 0.9
    do_gt_nmcs = True
    type = ""

