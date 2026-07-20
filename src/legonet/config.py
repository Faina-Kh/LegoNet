"""Shared mutable runtime configuration for LegoNet experiments."""

from __future__ import annotations

import enum


class DrawProperties:
    """Drawing options used by evaluation visualizations."""

    POINT_RADIUS: int = 5
    LINE_WIDTH: int = 5
    DRAW_MAPS: bool = False
    save_img_path: str = ''
    maps_path: str = ''

class NetworkType(enum.Enum):
    """Internal network families used for runtime dispatch."""

    detection = 1
    per_image_estimation_keypoints = 2
    detection_and_estimation = 3
    per_image_estimation_regression = 4


class General:
    """Experiment-wide runtime settings populated by the CLI."""

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
    """Detection thresholds and anchor settings."""

    NMS_THRESHOLD = 0.3 #0.5
    iou_threshold = 0.5
    min_score = 0.7 #0.05 #0.7
    iou_threshold_list = [0.3, 0.5, 0.7]
    min_score_list = []
    change_anchors = False
    ratios=None

class AttributeEstimation:
    """Image- and object-attribute estimation settings."""

    crops_size = [640,640]
    map_1_R = (6, 6)
    map_2_R = (5, 5)
    map_3_R = (5, 3)
    map_4_R = (3, 5)
    map_5_R = (3, 3)

    calc_det_performance = False
    do_nmcs = False
    estimate_type = "" #'reg_fpn_p3_p7_min_sig' #('reg_fpn_p3' #) #'withKeyPoints'
    per_object_count_target = "matched_gt"
    inter_losses = True
    num_of_pyr_levels = 5

class Detect_and_Estimate:
    """Joint detection-and-estimation evaluation settings."""

    precision_thresh = 0.9
    do_gt_nmcs = True
    type = ""

