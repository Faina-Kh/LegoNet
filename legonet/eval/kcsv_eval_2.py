import numpy as np
import json
import os

import torch
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#plt.style.use('seaborn')


import tkinter

import util
from util import printf
import config

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def plot_precision_recall(saved_im_dir,AP, recalls, precisions):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title(("Precision-Recall Curve. AP@50 = {:.3f}").format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)
    ax.set(xlabel='recall', ylabel='precision')
    ax.grid()
    image_path = os.path.join(saved_im_dir, "Precision-Recall Curve_AP " + str(AP) + ".jpg")
    plt.savefig(image_path)
    plt.close(image_path)


def _get_detections(generator, model, dataloader, sampler, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    model.eval()

    with torch.no_grad():

        for iter_num, data in enumerate(dataloader):  # for index in range(len(dataset)):
            #print(iter_num)
            scale = data['scale']
            image = data['img'].clone().detach()  # .permute(0, 2, 3, 1)

            # run network
            group_idx = sampler.groups[iter_num]
            #img_id = generator.image_ids[group_idx[0]]

            # detection_outputs, counting_outputs, corrected_counting_anns = legonet([image.cuda().float(), [data['bbox_annot'], data['points_annot']],
            #      torch.tensor(group_idx)])
            if config.General.NETWORK_TYPE == config.NetworkType.detection:
                detection_outputs = model([image.to(config.General.device).float(),
                                           data['bbox_annot'].to(config.General.device)])
                    #[image.cuda().float(), [data['bbox_annot'], None], torch.tensor(group_idx), False])  # image.cuda().float().unsqueeze(dim=0))
            else:

                if config.Detect_and_Estimate.type == "both_for_roots_2":
                    if 'points_annot' in data.keys():
                        detection_outputs, count_outputs, count_sample, relevant_points, crops_orig_boxes = \
                            model([image.to(config.General.device).float(),
                                   [data['bbox_annot'], data['points_annot']],
                                   torch.tensor(group_idx), True])

                    else:
                        continue

                else:
                    detection_outputs, count_outputs, count_sample, relevant_points, crops_orig_boxes = \
                        model([image.to(config.General.device).float(), [data['bbox_annot'], data['points_annot']],
                               torch.tensor(group_idx)])  # image.cuda().float().unsqueeze(dim=0))


            scores, labels, boxes = detection_outputs

            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                # scores_sort = np.argsort(-scores)[:max_detections]
                scores_sort = np.argsort(-scores)

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(generator.num_classes()):
                    all_detections[group_idx[0]][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(generator.num_classes()):
                    all_detections[group_idx[0]][label] = np.zeros((0, 5))

            print('{}/{}'.format(group_idx[0] + 1, len(generator)), end='\r')

    return all_detections


def  _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            anns=annotations[0]
            all_annotations[i][label] = anns[anns[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def _get_count_and_box_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_box_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]
    all_count_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            box_anns=annotations[0]  #[x1,y1,x2,y2,class,box_id]
            if len(annotations[1])>0:
                count_anns=np.array(annotations[1][0])
                all_count_annotations[i][label] = count_anns[count_anns[:, 1] == label, :].copy()

            else:
                count_anns=[]
                all_count_annotations[i][label] = []

            if len(box_anns) > 0:
                all_box_annotations[i][label] = box_anns[box_anns[:, 4] == label, :].copy()
            else:
                all_box_annotations[i][label] = []


        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_box_annotations,all_count_annotations



def evaluate(generator,
             dataloader_val,
             sampler_val,
             model,
             iou_threshold=[0.25, 0.5, 0.75],
             score_threshold=[0.01, 0.05, 0.5],
             max_detections=100,
             save_path=None,
             show_PR_curve = False
             ):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet       : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    if show_PR_curve:
        dims = [len(score_threshold), len(iou_threshold)]
        fig, axs = plt.subplots(dims[0], dims[1], sharex=True, sharey=True)



    # gather all annotations
    all_annotations = _get_annotations(generator)

    average_precisions_all = []

    for score_index in range(len(score_threshold)):

        score = score_threshold[score_index]
        # gather all detections
        all_detections = _get_detections(generator, model, dataloader_val, sampler_val,
                                         score_threshold = score,
                                         max_detections = max_detections,
                                         save_path=save_path)
        average_precisions = {}

        for iou_index in range(len(iou_threshold)):
            iou = iou_threshold[iou_index]
            for label in range(generator.num_classes()):
                false_positives = np.zeros((0,))
                true_positives = np.zeros((0,))
                scores = np.zeros((0,))
                num_annotations = 0.0

                for i in range(len(generator)):
                    detections = all_detections[i][label]
                    annotations = all_annotations[i][label]

                    ##################################################
                    if detections is None:
                        continue
                    #################################################

                    num_annotations += annotations.shape[0]
                    detected_annotations = []

                    for d in detections:
                        scores = np.append(scores, d[4])

                        if annotations.shape[0] == 0:
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)
                            continue

                        overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap = overlaps[0, assigned_annotation]

                        if max_overlap >= iou and assigned_annotation not in detected_annotations:
                            false_positives = np.append(false_positives, 0)
                            true_positives = np.append(true_positives, 1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)

                # no annotations -> AP for this class is 0 (is this correct?)
                if num_annotations == 0:
                    average_precisions[label] = 0, 0
                    continue

                # sort by score
                indices = np.argsort(-scores)
                false_positives = false_positives[indices]
                true_positives = true_positives[indices]

                # compute false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)

                # compute recall and precision
                recall = true_positives / num_annotations
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                # compute average precision
                average_precision = _compute_ap(recall, precision)
                average_precisions[label] = average_precision, num_annotations

                util.printf("mAP (score >= %.3f, IoU >= %.3f)\n",score, iou)
                for label in range(generator.num_classes()):
                    class_name = generator.label_to_name(label)
                    class_mAP = average_precisions[label][0]
                    util.printf("%s: %.4f\n", class_name, class_mAP)
                    average_precisions_all.append([iou, score, class_mAP])

                if show_PR_curve:
                    axs[score_index,iou_index].plot(recall, precision)
                    axs[score_index,iou_index].set_title("score:"+str(score)+ ", IoU:"+str(iou), fontsize=8)
                    #axs[score_index,iou_index].set(xlabel='recall', ylabel='precision')
                    axs[score_index, iou_index].tick_params(axis="x", labelsize=6)
                    axs[score_index, iou_index].tick_params(axis="y", labelsize=6)



    if show_PR_curve:
        plt.savefig(os.path.join(save_path, "thresh.jpg"))
        plt.show()
            # if (show_PR_curve):
            #     # draw precision recall curve
            #     demmi_recall = np.linspace(0,1,101, endpoint=True)
            #     R = demmi_recall
            #
            #     tp_sum = np.cumsum(true_positives).astype(dtype=np.float)
            #     fp_sum = np.cumsum(false_positives).astype(dtype=np.float)
            #     for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            #         tp = np.array(tp)
            #         fp = np.array(fp)
            #         nd = tp.size
            #
            #         pr = tp / (fp + tp + np.spacing(1))
            #         q = np.zeros(R.size)
            #
            #
            #     pr = pr.tolist();
            #     q = q.tolist()
            #
            #     for i in range(nd - 1, 0, -1):
            #         if pr[i] > pr[i - 1]:
            #             pr[i - 1] = pr[i]
            #
            #     precision = np.array(q)
            #
            #     saved_im_dir = os.path.join(os.getcwd(), 'results')
            #
            #     plot_precision_recall(saved_im_dir,  map, demmi_recall, precision)
            #
            #     plt.plot(recall, precision)
            #     plt.xlabel('recall')
            #     plt.ylabel('precision')
            #     plt.show()
            #     plt.savefig('detect_curve')

    return average_precisions_all #average_precisions


def evaluateMAP(generator,
                dataloader_val,
                sampler_val,
                model,
                iou_threshold=[0.3, 0.5, 0.7, 0.9],
                score_threshold=0.05,
                max_detections=1000,
                save_path=None,
                generate_PR_curve = False
                ):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all annotations
    all_annotations = _get_annotations(generator)

    score = score_threshold

    # gather all detections
    all_detections = _get_detections(generator, model, dataloader_val, sampler_val,
                                         score_threshold = score,
                                         max_detections = max_detections,
                                         save_path=save_path)
    average_precisions = {}

    output = {}
    for iou in iou_threshold:
        output[iou] = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(len(generator)):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations

        class_name = []
        class_mAP  = []
        for label in range(generator.num_classes()):
            class_name.append(generator.label_to_name(label))
            class_mAP.append(average_precisions[label][0])

        output[iou]["class_name"] = class_name
        output[iou]["class_mAP"]  = class_mAP

        if generate_PR_curve:
            class_map_str = "{:.3f}".format(class_mAP[0])
            plt.plot(recall, precision, label=class_name[0] + " IoU = " + str(iou) + " mAP = " + class_map_str )

    if generate_PR_curve:
        plt.legend(loc="best")
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(config.General.experiment_path,"PR_curve.png"))

    return output

def evaluateMAP_simple(generator,
                dataloader_val,
                sampler_val,
                model,
                iou_threshold=0.5,
                score_threshold=0.05,
                max_detections=1000,
                generate_PR_curve=False
                ):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all annotations
    all_annotations = _get_annotations(generator)

    score = score_threshold

    # gather all detections
    all_detections = _get_detections(generator, model, dataloader_val, sampler_val,
                                         score_threshold = score,
                                         max_detections = max_detections)
    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            ########################################################################
            scores_temp = np.zeros((0,))
            if detections is None:
                continue

            for d in detections:
                scores_temp = np.append(scores_temp, d[4])

            indices = np.argsort(-scores_temp)

            detections = detections[indices,:]

            ########################################################################

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    mAP = np.zeros(generator.num_classes())
    for label in range(generator.num_classes()):
        mAP[label] = average_precisions[label][0]

    if generate_PR_curve:
        plt.plot(recall, precision)
        plt.title(f'mAP={(np.mean(mAP)):.3f}')
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(config.DrawProperties.save_img_path,"PR_curve_objects.png"))



    # precision_idx=  np.argsort(-precision)
    # precision2=precision[precision_idx]
    # recall2=recall[precision_idx]
    #
    # precision_threshold=[0.75, 0.9, 0.95, 0.99]
    # for th in precision_threshold:
    #     relevant_idx=np.nonzero(precision2>th)[-1]
    #     pr= precision2[relevant_idx]
    #     rc=recall2[relevant_idx]
    #
    #     print('th={}, pr={}, rc={}'.format(th, pr, rc))


    return np.mean(mAP), np.mean(precision), np.mean(recall) #np.mean(mAP), precision, recall #  None, None precision, recall


def evaluate_double_detection(dataset_val_with_counts,
                            generator,
                            dataloader_val,
                            sampler_val,
                            model,
                            iou_threshold=0.5,
                            score_threshold=0.05,
                            max_detections=1000,
                            verbose = True,
                            assign_parts_to_obj = True,
                            generate_PR_curve=False
                            ):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all annotations
    all_annotations = _get_annotations(generator)

    score = score_threshold

    # gather all detections
    all_detections = _get_detections(generator, model, dataloader_val, sampler_val,
                                         score_threshold = score,
                                         max_detections = max_detections)
    average_precisions = {}

    tp = []
    fp = []

    _, all_count_annotations = _get_count_and_box_annotations(dataset_val_with_counts)
    per_img_avg_pred_count = []
    per_img_avg_gt_count = []
    per_imge_rel_error = []
    ####################################################################################################
    for iter_num, data in enumerate(dataloader_val):
        group_idx = sampler_val.groups[iter_num]
        img_id = generator.image_ids[group_idx[0]]

        assert generator.img_info[img_id]['name'] == dataset_val_with_counts.img_info[img_id]['name']
        if verbose:
            printf("#######################################################################################\n")
            printf("image: %s\n", generator.img_info[img_id]['name'])
            printf("#######################################################################################\n")

        gt_counts_temp = all_count_annotations[img_id][0]
        gt_count = np.sum(gt_counts_temp[:, 0]) / gt_counts_temp.shape[0]
        per_img_avg_gt_count.append(gt_count)

        obj_detections = all_detections[img_id][0]
        part_detections = all_detections[img_id][1]

        if assign_parts_to_obj:
            if verbose:
                print('assign_parts_to_obj')
            count_per_obj = []
            for obj_box in obj_detections:
                count=0

                for p in part_detections:
                    w = p[2] - p[0]
                    h = p[3] - p[1]
                    px = p[0] + 0.5*w
                    py = p[1] + 0.5*h

                    if px<= obj_box[2] and px>= obj_box[0] and py<= obj_box[3] and py>= obj_box[1]:
                        count+=1

                if count>0:
                    count_per_obj.append(count)



        if len(obj_detections) > 0:

            if assign_parts_to_obj:
                if len(count_per_obj) > 0:
                    pred_count = np.mean(count_per_obj)
                else:
                    pred_count = -1

            else:
                pred_count = len(part_detections)/len(obj_detections)


            if not pred_count == -1:
                per_img_avg_pred_count.append(pred_count)
                per_imge_rel_error.append(np.abs(pred_count - gt_count) / gt_count)
                if verbose:
                    printf("avg_gt_count: %.2f | avg_pred_count: %.2f | rel_error: %.3f\n",
                           gt_count, pred_count, per_imge_rel_error[-1])

            else:
                if verbose:
                    print("Couldn't assign parts to objects")

        else:
            if verbose:
                print('No detected objects')

    avg_error = np.mean(per_imge_rel_error)
    if verbose:
        print("=====================================================================================================\n")
        print("Avg rel error: ", avg_error)


        print("=====================================================================================================\n")
        print('Per image gt stats')
        print('avg of image gt averages = ', np.mean(per_img_avg_gt_count))
        print('var of image gt averages = ', np.var(per_img_avg_gt_count))
        #print("per image gt avg: ", np.round(per_img_avg_gt_count, 2))
        print()

        print("=====================================================================================================\n")
        print('Per image pred stats')
        print('avg of image pred averages = ', np.mean(per_img_avg_pred_count))
        print('var of image pred averages = ', np.var(per_img_avg_pred_count))
        # print("per image pred avg: ", np.round(per_img_avg_pred_count, 2))
        print()

    return avg_error
    ###################################################################################################

    #
    # get_gt = True
    # for label in range(generator.num_classes()):
    #     false_positives = np.zeros((0,))
    #     true_positives = np.zeros((0,))
    #     scores = np.zeros((0,))
    #     num_annotations = 0.0
    #
    #     #for i in range(len(generator)):
    #     for iter_num, data in enumerate(dataloader_val):
    #         group_idx = sampler_val.groups[iter_num]
    #         img_id = generator.image_ids[group_idx[0]]
    #         if get_gt:
    #             gt_counts_temp = all_count_annotations[img_id][0]
    #             per_img_avg_gt_count.append(np.sum(gt_counts_temp[:, 0]) / gt_counts_temp.shape[0])
    #
    #         detections = all_detections[img_id][label] #all_detections[i][label]
    #         annotations = all_annotations[img_id][label]  #all_annotations[i][label]
    #
    #         num_annotations += annotations.shape[0]
    #         detected_annotations = []
    #
    #         ########################################################################
    #         scores_temp = np.zeros((0,))
    #         for d in detections:
    #             scores_temp = np.append(scores_temp, d[4])
    #
    #         indices = np.argsort(-scores_temp)
    #
    #         detections = detections[indices,:]
    #
    #         ########################################################################
    #
    #
    #         for d in detections:
    #             scores = np.append(scores, d[4])
    #
    #             if annotations.shape[0] == 0:
    #                 false_positives = np.append(false_positives, 1)
    #                 true_positives = np.append(true_positives, 0)
    #                 continue
    #
    #             overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
    #             assigned_annotation = np.argmax(overlaps, axis=1)
    #             max_overlap = overlaps[0, assigned_annotation]
    #
    #             if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
    #                 false_positives = np.append(false_positives, 0)
    #                 true_positives = np.append(true_positives, 1)
    #                 detected_annotations.append(assigned_annotation)
    #             else:
    #                 false_positives = np.append(false_positives, 1)
    #                 true_positives = np.append(true_positives, 0)
    #
    #     get_gt = False
    #
    #
    #     # no annotations -> AP for this class is 0 (is this correct?)
    #     if num_annotations == 0:
    #         average_precisions[label] = 0, 0
    #         continue
    #
    #     # sort by score
    #     indices = np.argsort(-scores)
    #     false_positives = false_positives[indices]
    #     true_positives = true_positives[indices]
    #
    #     # compute false positives and true positives
    #     false_positives = np.cumsum(false_positives)
    #     true_positives = np.cumsum(true_positives)
    #
    #     # compute recall and precision
    #     recall = true_positives / num_annotations
    #     precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    #
    #     # compute average precision
    #     average_precision = _compute_ap(recall, precision)
    #     average_precisions[label] = average_precision, num_annotations
    #
    #     if true_positives.size != 0:
    #         tp.append(np.max(true_positives))
    #     else:
    #         tp.append(0.0)
    #
    #     if false_positives.size !=0:
    #         fp.append(np.max(false_positives))
    #     else:
    #         fp.append(0.0)
    #
    # count_tp = 0
    # count_all = 0
    # if tp[0] != 0:
    #     count_tp = tp[1]/tp[0]
    #
    # if (tp[0] + fp[0]) != 0:
    #     count_all = (tp[1] + fp[1])/(tp[0] + fp[0])
    #
    # # count all : # all class 1/ # all class 0
    # # count tp: # tp class 1/ # tp class 0
    #
    # return avg_error, count_tp, count_all, tp, fp
