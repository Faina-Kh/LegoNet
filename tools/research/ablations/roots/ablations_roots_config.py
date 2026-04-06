import os
import config
import numpy as np



def ablations(args):
    config.roots_ablations.freeze_all_except_find_2 = False

    if (args.network_type == "detection" and args.dataset_name == 'roots') or args.network_type == "both_for_roots_2":
        config.Detection.change_anchors = True
        config.Detection.ratios = np.array(
            [0.5, 1, 3])  # np.array([0.5, 1, 3]) #np.array([0.5, 1, 4]) #np.array([0.5, 1, 2])
        print("Detection.ratios:", config.Detection.ratios)

    if args.network_type == "detection":
        args.eval_detection_params = False
        config.Detection.iou_threshold_list = [0.3, 0.5, 0.7]

    args.evaluate_both = True  # relevant to validation, not train