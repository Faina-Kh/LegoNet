import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from legonet import utils
from legonet.utils import printf
from legonet import config




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


def plot_PR_curve(recall, precision, ap, save_path, plots_name=""):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.99)
    plt.fill_between(recall, precision, step='post', color='b', alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(ap))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_path = os.path.join(save_path + '\\'+ plots_name) #'Points_PR_curve.png')
    plt.savefig(plot_path)
    plt.close(plot_path)


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

                if config.Detect_and_Estimate.type == "per_object_attributes" or config.Detect_and_Estimate.type == "per_object_attributes_multibranch":
                    if 'points_annot' in data.keys():
                        detection_outputs, count_outputs, count_sample, relevant_points, crops_orig_boxes = \
                            model([image.to(config.General.device).float(),
                                   [data['bbox_annot'], data['points_annot']],
                                   torch.tensor(group_idx)]) #, True])

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


def evaluate_detection_params(generator,
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
        fig, axs = plt.subplots(dims[0], dims[1], sharex=True, sharey=True, squeeze=False)


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

                    if detections is None:
                        continue

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

                # no annotations -> AP for this class is 0
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

                utils.printf("mAP (score >= %.3f, IoU >= %.3f)\n",score, iou)
                for label in range(generator.num_classes()):
                    class_name = generator.label_to_name(label)
                    class_mAP = average_precisions[label][0]
                    utils.printf("%s: %.4f\n", class_name, class_mAP)
                    average_precisions_all.append([iou, score, class_mAP])

                if show_PR_curve:
                    axs[score_index,iou_index].plot(recall, precision)
                    axs[score_index,iou_index].set_title("score:"+str(score)+ ", IoU:"+str(iou), fontsize=8)
                    axs[score_index,iou_index].set(xlabel='recall', ylabel='precision')
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
    class_precisions = []
    class_recalls = []
    pr_curve_recall = np.zeros((0,))
    pr_curve_precision = np.zeros((0,))

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

        if scores.shape[0] == 0:
            average_precisions[label] = 0, num_annotations
            class_precisions.append(0.0)
            class_recalls.append(0.0)
            continue

        # compute false positives and true positives
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        pr_curve_recall = recall
        pr_curve_precision = precision
        class_precisions.append(float(precision[-1]))
        class_recalls.append(float(recall[-1]))

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    mAP = np.zeros(generator.num_classes())
    for label in range(generator.num_classes()):
        mAP[label] = average_precisions[label][0]

    if generate_PR_curve:
        plt.plot(pr_curve_recall, pr_curve_precision)
        plt.title(f'mAP={(np.mean(mAP)):.3f}')
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(config.General.files_path,"PR_curve_objects.png"))



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


    final_precision = float(np.mean(class_precisions)) if class_precisions else 0.0
    final_recall = float(np.mean(class_recalls)) if class_recalls else 0.0

    return np.mean(mAP), final_precision, final_recall #np.mean(precision), np.mean(recall) #np.mean(mAP), precision, recall #  None, None precision, recall

