import os
import config
import numpy as np


def ablations(args):

    # points model

    if (args.dataset_name == "131_wheat_spikes_and_spikelets" or args.dataset_name == "grapes" or
            args.dataset_name == "roots"):
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

    config.AttributeEstimation.do_nmcs = True
    config.AttributeEstimation.inter_losses = True
    config.AttributeEstimation.num_of_pyr_levels = 5

    config.Detect_and_Estimate.single_backbone = False
    config.Detect_and_Estimate.crop_from_Pi = False
    config.Detect_and_Estimate.cancel_nms_in_train = False


    config.DrawProperties.DRAW_MAPS = True

    config.Detection.min_score = 0.7  # 0.95 #0.5 #0.05
    config.Detection.iou_threshold = 0.5  # 0.5 #[0.3, 0.5, 0.7, 0.9, 0.95] #0.5
    config.Detection.min_score_list = [0.7]

    args.eval_detection_params = True
    config.Detection.iou_threshold_list = [0.3, 0.5, 0.7]

    config.Detection.NMS_THRESHOLD = 0.3
    config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False
    config.Detection.USE_PERFECT_DETECTION_MODE = False

    config.Detection.BBOX_ADJUSTMENT_RATIO = 1  # 1.25

    config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING = False
    config.Detection.BBOX_SCALING_FOR_COUNTING_AUGMENTATION = np.linspace(start=0.75, stop=1.25, num=3)

    # config.detect_and_count.choose_by_IoUandPrc_Flag = False
    # config.detect_and_count.precision_thresh = 0.3
    # config.detect_and_count.do_gt_nmcs = False

