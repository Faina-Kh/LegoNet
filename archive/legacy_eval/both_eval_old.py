from pycocotools.cocoeval import COCOeval
import json
import torch

import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import PIL
from PIL import Image
from torchvision import transforms

from util import printf
from legonet.eval.counting_eval import SumOfAbsDifferences
from legonet.eval.kcsv_eval_2 import _get_count_and_box_annotations, compute_overlap, _compute_ap
from legonet.eval.count_detection_eval import detection_evaluation, calc_recall_precision_ap
from legonet.myDataloader import UnNormalizer

import config
import copy
from itertools import compress

unnormalize = UnNormalizer()


def _get_detections(detection_outputs, scale):
    scores, labels, boxes = detection_outputs
    boxes = boxes.cpu().numpy()

    # correct boxes for image scale
    boxes /= scale

    return boxes

def plot_RP_curve(recall, precision, ap, save_path):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.99)
    plt.fill_between(recall, precision, step='post', color='b', alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_path = os.path.join(save_path + '\\points_RP_curve.png')
    plt.savefig(plot_path)
    plt.close(plot_path)

def find_points_in_bbox(img, point_anns, bbox_pred, scale):

    points=[]

    # for drawing
    im = img.cpu().clone().detach()
    im = np.array(255 *unnormalize(im))
    im[im < 0] = 0
    im[im > 255] = 255
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)

    for p in point_anns:
        p['x'] = p['x'] * scale
        p['y'] = p['y'] * scale

        cv2.circle(im, (int(p['x']), int(p['y'])), radius=3, color=(0, 0, 255), thickness=2)


    for b in range(bbox_pred.shape[0]):
        p_x = []
        p_y = []
        box_x1, box_y1, box_x2, box_y2 = bbox_pred[b]
        cv2.rectangle(im, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), color=(0, 0, 255), thickness=2)

        for p in point_anns:
            if p['x']<=box_x2 and p['x']>=box_x1 and p['y']<=box_y2 and p['y']>=box_y1:
                p_x.append(p['x'])
                p_y.append(p['y'])

        points.append({'x':p_x, 'y':p_y})

    # cv2.imshow('img', im)
    # cv2.waitKey(0)

    return points

def visualize_images(count_outputs, count_sample, image_name, imgToVis, draw_path):

    # Draw GT activations:
    img_copy = imgToVis.copy()
    background = img_copy.convert("RGBA")
    BG_w, BG_h = background.size

    background_2=img_copy.convert('L')   #convert image to monochrome
    background_2.save(draw_path + '/' + image_name + '_background.png')
    background_2 = Image.open(draw_path + '/' + image_name + '_background.png')
    background_2=background_2.convert("RGBA")

    anno = count_sample.copy() #.cpu().numpy().copy()

    plt.imsave(draw_path + '/' + image_name + '_anno.png', anno)
    gt_anns = Image.open(draw_path + '/' + image_name + '_anno.png')
    gt_anns = gt_anns.resize((BG_w, BG_h), Image.ANTIALIAS)
    gt_anns.save(draw_path + '/' + image_name + '_anno.png')

    alphaBlended = Image.blend(gt_anns, background_2, 0.6)
    alphaBlended.save(draw_path + '/' + image_name + '_Blended_GT.png')

    # Relu map #######################################################################################################

    plt.imsave(draw_path + '/' + image_name + '_Relu.png', count_outputs.cpu())
    relu_anns = Image.open(draw_path + '/' + image_name + '_Relu.png')

    relu_anns = relu_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
    relu_anns.save(draw_path + '/' + image_name + '_Relu.png')

    alphaBlended_relu = Image.blend(relu_anns, background_2.convert('RGBA'), 0.6)
    alphaBlended_relu.save(draw_path + '/' + image_name + '_Blended_Relu.png')

    os.remove(draw_path + '/' + image_name + '_background.png')
    os.remove(draw_path + '/' + image_name + '_anno.png')
    os.remove(draw_path + '/' + image_name + '_Relu.png')

def nmcs(predicted_boxes, relevant_points):

    non_supressed_indices = [True for i in range(predicted_boxes.shape[0])]

    for i in range(predicted_boxes.shape[0]):
        current_points = relevant_points[i]

        if len(current_points["x"]) == 0:  # no relevant points
            continue

        for j in range(predicted_boxes.shape[0]):

            candidate_points = relevant_points[j]

            if i == j or len(candidate_points["x"]) == 0 or non_supressed_indices[j] == False:
                continue
            else:
                common_points_count = 0
                for m in range(len(candidate_points["x"])):
                    x1, y1 = candidate_points["x"][m], candidate_points["y"][m],
                    for n in range(len(current_points["x"])):
                        x2, y2 = current_points["x"][n], current_points["y"][n],
                        if x1 == x2 and y1 == y2:
                            common_points_count += 1
                if common_points_count == len(candidate_points["x"]):
                    non_supressed_indices[j] = False

    return non_supressed_indices

def choose_boxes_by_IoUandPrc(detections, annotations, d_scores):
    annotations = annotations[0,:,:4]

    iou_threshold = config.Detection.iou_threshold
    precision_thresh = config.detect_and_count.precision_thresh

    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    num_annotations = annotations.shape[0]
    detected_annotations = []

    indices = np.argsort(-d_scores.cpu())
    detections = detections[indices, :]

    for d in detections:
        if annotations.shape[0] == 0:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)
            continue

        overlaps = compute_overlap(np.expand_dims(d.cpu(), axis=0), annotations.numpy())
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
        mAP = 0
    else:

        # sort by score
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        mAP = _compute_ap(recall, precision)

        # sort by precision
        precision_idx= np.argsort(-precision)
        precision2=precision[precision_idx]
        recall2=recall[precision_idx]

        relevant_idx=np.where(precision2>precision_thresh)[0]
        if len(relevant_idx) >0:
            relevant_idx=relevant_idx[-1]

            pr= precision2[relevant_idx]
            rc=recall2[relevant_idx]
            print('mAP = {:0.3f}, th={}, pr={}, rc={}\n'.format(mAP, precision_thresh, pr, rc))
            detections=detections[:(relevant_idx+1)]

            return detections

        else:
            print('No relevant detections...\n')
            return []

def view_points_on_img(img, point_anns):

    for p in point_anns:
        cv2.circle(img, (int(p['x']), int(p['y'])), radius=3, color=(0, 0, 255), thickness=2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

##########################################################################################
def image_output_shape(image_shape, pyramid_level=3):
    return (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)

def images_ratios(image_shape, output_shape):
    return output_shape / np.array(image_shape[:2])


def create_gausian_mask(center_point, nCols, nRows, q=99, radius=(5, 5)):
    '''
    create_gausian_mask creates a gaussian mask to be used as GT annotations for the detection-based counter
    :param center_point:
    :param nCols:
    :param nRows:
    :param q:
    :param s:
    :param radius:
    :return:
    '''
    s = 3
    # if (s >= radius[0]):
    #     s = 1
    x = np.tile(range(nCols), (nRows, 1))
    y = np.tile(np.reshape(range(nRows), (nRows, 1)), (1, nCols))

    x2 = (((x - np.round(center_point[0])) * s) / radius[0]) ** 2
    y2 = (((y - np.round(center_point[1])) * s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return p


def compute_keypoints_targets_multi_maps(image_shape, annotations_points_centers_a, radius=(5, 5), pyramid_level=3):
    # resize transformed-image and annotations
    import copy
    annotations_points_centers = copy.deepcopy(annotations_points_centers_a)
    # here we should resize image too and then check it with the annotations
    output_shape = image_output_shape(image_shape[2:], pyramid_level=pyramid_level)
    image_ratio = images_ratios(image_shape[2:], output_shape)

    if len(annotations_points_centers) == 0:
        return [np.zeros(output_shape)]

    per_img_anns = []
    img_num = len(annotations_points_centers)
    for i in range(img_num):
        current_points_a = annotations_points_centers[i]  # [N, [x, y, points class, bbox_id]]
        current_points = np.array([current_points_a['x'], current_points_a['y']])
        current_points[0] = current_points[0] * image_ratio[0]
        current_points[1] = current_points[1] * image_ratio[1]
        annotations = np.zeros(output_shape)
        for j in range(current_points[0].shape[0]):
            gaussian_map = create_gausian_mask((current_points[0,j],current_points[1,j]), output_shape[1], output_shape[0],
                                                    radius=radius)
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)

        if np.isnan(annotations).any():
            raise ("nan was found")

        per_img_anns.append(annotations)

    return per_img_anns


#########################################################################################


def eval(dataset, dataloader, sampler, model, verbose=True, to_draw=True):

    model.eval()

    if to_draw:
        draw_path = os.path.join(config.General.experiment_path, 'vis')
        if not os.path.exists(draw_path):
            os.makedirs(draw_path)

    # gather all annotations, per image, per label
    all_box_annotations, all_count_annotations = _get_count_and_box_annotations(dataset)

    with torch.no_grad():

        # for counting
        T, P = [], []

        all_predicted_counts = []

        all_crops_GT_counts = []
        crops_abs_diff = []
        crops_rel_error = []

        all_orig_GT_counts = []
        orig_abs_diff = []
        orig_rel_error = []

        all_data_gt_count = []
        gt_objects_withGTpoints=0
        found_orig_objects = 0
        FP=0
        predicted_counts_any_crop = []
        matched_without_gt_points = 0
        crops_without_gt_points = 0

        per_im_avg = []
        for iter_num, data in enumerate(dataloader):

            # get per image stats
            crops_count_GT = []
            orig_count_GT = []
            count_pred = []

            image = data['img'].clone().detach()
            scale= data['scale']

            group_idx=sampler.groups[iter_num]
            img_id = dataset.image_ids[group_idx[0]]

            image_name = dataset.img_info[img_id]['name']
            image_path = os.path.join(dataset.base_dir, image_name)

            # get image annotations - bboxes and counts
            box_annotations_temp = all_box_annotations[img_id][0]
            gt_counts_temp = all_count_annotations[img_id][0]

            per_im_avg.append(np.sum(gt_counts_temp[:,0])/gt_counts_temp.shape[0])

            box_annotations = []
            gt_counts=[]
            ###############################################################
            # print('image_name:', image_name)

            for i in range(len(box_annotations_temp)):
                b=box_annotations_temp[i]
                id=box_annotations_temp[i][5]

                #filtering the annotations - prevents having gt boxes without gt points:
                for g in gt_counts_temp:
                    if g[2] == id:
                        gt_counts.append(g)
                        box_annotations.append(torch.tensor(b).unsqueeze(dim=0))
                        break


            if len(box_annotations)==0:
                if verbose:
                    printf(
                        "##############################################################################################\n")
                    printf("image: %s\n", image_name)
                    printf(
                        "##############################################################################################\n")
                    printf("No gt points in any gt box...\n")
                    print()
                continue

            #from list to tensor:
            if len(box_annotations)>1:
                box_annotations = torch.cat(box_annotations, dim=0)
            else:
                box_annotations=box_annotations[0]


            ##############################################################
            # get stats - count gt boxes with gt points
            for c in gt_counts:
                all_data_gt_count.append(c[0])
                gt_objects_withGTpoints += 1

            # run the network
            detection_outputs, count_outputs, count_sample, relevant_points, crops_orig_boxes = \
                model([image.to(config.General.device).float(), [data['bbox_annot'], data['points_annot']], torch.tensor(group_idx), True]) #image.cuda().float().unsqueeze(dim=0))

            if count_outputs is not None:
                for c_out in count_outputs[0]:
                    predicted_counts_any_crop.append(c_out.cpu().numpy())
            ###################################################################################################################################################
            # detection_outputs - outputs of the detection part (based on module where), after filtering by nms and min score
            # count_outputs - prediction of counting per box from detection_outputs
            # count_sample - has the crop in its 'img' key. In training it has also 'points_annot' key that holds the gt annotations per crop - relevant
            # for evaluation during training.
            ###################################################################################################################################################

            # original image
            orig_img = Image.open(image_path)

            # bbox_pred - all predicted boxes - with or without points in it - rescaled to the orig img size
            if config.Detection.USE_PERFECT_DETECTION_MODE:
                bbox_pred = data['bbox_annot'].clone()
                bbox_pred = bbox_pred.squeeze(dim=0)[:,:4].cpu().numpy()
                bbox_pred /= scale
            else:
                bbox_pred = _get_detections(detection_outputs, scale)

            if bbox_pred is None:
                if verbose:
                    printf(
                        "##############################################################################################\n")
                    printf("image: %s\n", image_name)
                    printf(
                        "##############################################################################################\n")
                    printf("Image has no predicted boxes...\n")
                    print()

                continue

            point_anns = dataset.image_data_points_location[image_name]  # data from kcsv file

            if to_draw:
                draw = PIL.ImageDraw.Draw(orig_img)

                gt_boxes = data['bbox_annot'][0]
                for i in range(gt_boxes.shape[0]):
                    x1 = gt_boxes[i, 0].numpy() / scale
                    y1 = gt_boxes[i, 1].numpy() / scale
                    x2 = gt_boxes[i, 2].numpy() / scale
                    y2 = gt_boxes[i, 3].numpy() / scale
                    draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)

                for b in range(bbox_pred.shape[0]):
                    x1 = bbox_pred[b, 0]
                    y1 = bbox_pred[b, 1]
                    x2 = bbox_pred[b, 2]
                    y2 = bbox_pred[b, 3]
                    draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=config.DrawProperties.LINE_WIDTH)

                for p in point_anns:
                    r = config.DrawProperties.POINT_RADIUS
                    draw.ellipse((p['x']-r, p['y']-r, p['x']+r, p['y']+r), fill="black", width=config.DrawProperties.LINE_WIDTH)

                #orig_img.show(title=image_name)
                orig_img.save(os.path.join(draw_path, image_name))

            ############################################################################################################
            # Preparing the evaluation of the counting results per crop - doesn't depend on the detection performance -
            # we only need to find and evaluate the crops that include gt points - otherwise we'll compare
            # the predicted count to 0, probably since the points weren't annotated
            ############################################################################################################

            if count_sample is not None:  # count_sample: corrected gt - has gt annotations for the crops
                if 'points_annot' in count_sample.keys():
                    # evaluating during training
                    crops_count_GT = np.array(count_sample['points_annot'][0].cpu())
                    all_crops_GT_detections_maps = count_sample['points_annot'][5]
                    all_predicted_detection_maps = count_outputs[6]
                    for b in range(count_outputs[0].shape):
                        count_pred.append(count_outputs[0][b].cpu().item())

                else:
                    if config.Counting.counting_type == 'withKeyPoints':
                        # filtering the predictions - need to find predicted boxes (and maps) with gt points in them
                        orig_predicted_detection_maps = [count_outputs[1],count_outputs[2],count_outputs[3],count_outputs[4],count_outputs[6]] #count_outputs[6]
                        all_predicted_detection_maps = []
                        all_predicted_detection_maps_toDraw=[]

                    all_crops_GT_detections_maps = []

                    adjusted_crops_orig_boxes = []
                    relevant_points = []
                    count_sample['points_annot']=[]
                    if not isinstance(bbox_pred, list):
                        for b in range(bbox_pred.shape[0]):

                            count_pred.append(count_outputs[0][b].cpu().item())
                            # ToDo - fix code duplication (same as in model3\\getitem)
                            adjusted_crops_orig_boxes.append(crops_orig_boxes[b])

                            x1 = bbox_pred[b, 0]
                            y1 = bbox_pred[b, 1]
                            x2 = bbox_pred[b, 2]
                            y2 = bbox_pred[b, 3]

                            points_of_current_crop = {}
                            current_count = 0
                            for p in point_anns:
                                if p['x'] <= x2 and p['x'] >= x1 and p['y'] <= y2 and p['y'] >= y1:
                                    if len(points_of_current_crop)==0:
                                        points_of_current_crop['x']=[]
                                        points_of_current_crop['y'] = []

                                    points_of_current_crop['x'].append(p['x'])
                                    points_of_current_crop['y'].append(p['y'])
                                    current_count +=1

                            if current_count == 0:
                                crops_without_gt_points +=1
                                continue

                            # rescale points (from orig coordinates) to the crop (the 'new image')
                            points_of_current_crop['x'] = points_of_current_crop['x'] - (
                                        x1 * np.ones(len(points_of_current_crop['x'])))
                            points_of_current_crop['y'] = points_of_current_crop['y'] - (
                                        y1 * np.ones(len(points_of_current_crop['y'])))
                            scale_x = config.Counting.crops_size[0] / (x2 - x1)
                            scale_y = config.Counting.crops_size[1] / (y2 - y1)
                            points_of_current_crop['x'] = points_of_current_crop['x'] * scale_x
                            points_of_current_crop['y'] = points_of_current_crop['y'] * scale_y

                            if to_draw:
                                points_to_view = []
                                for i in range(current_count):
                                    points_to_view.append({'x': points_of_current_crop['x'][i], 'y': points_of_current_crop['y'][i]})

                                relevant_points.append(points_to_view)

                            crops_count_GT.append(current_count)

                            if config.Counting.counting_type == 'withKeyPoints':
                                # generate gt gaussian maps for the crop
                                annotations_group_points_center = [points_of_current_crop]

                                annotation_map_1 = compute_keypoints_targets_multi_maps(count_sample['img'].shape,
                                                                                        annotations_group_points_center,
                                                                                        radius=config.Counting.map_1_R)
                                annotation_map_2 = compute_keypoints_targets_multi_maps(count_sample['img'].shape,
                                                                                        annotations_group_points_center,
                                                                                        radius=config.Counting.map_2_R)
                                annotation_map_3 = compute_keypoints_targets_multi_maps(count_sample['img'].shape,
                                                                                        annotations_group_points_center,
                                                                                        radius=config.Counting.map_3_R)
                                annotation_map_4 = compute_keypoints_targets_multi_maps(count_sample['img'].shape,
                                                                                        annotations_group_points_center,
                                                                                        radius=config.Counting.map_4_R)
                                annotation_map_5 = compute_keypoints_targets_multi_maps(count_sample['img'].shape,
                                                                                        annotations_group_points_center, radius=config.Counting.map_5_R)


                                count_sample['points_annot'].append([[current_count],
                                                         annotation_map_1, annotation_map_2, annotation_map_3,
                                                         annotation_map_4, annotation_map_5])

                                # keep predictions and gt only for those that have gt points
                                all_crops_GT_detections_maps.append(count_sample['points_annot'][-1][5][0])
                                all_predicted_detection_maps.append(orig_predicted_detection_maps[-1][b])
                                if to_draw:
                                    all_predicted_detection_maps_toDraw.append([
                                        orig_predicted_detection_maps[0][b], orig_predicted_detection_maps[1][b],
                                        orig_predicted_detection_maps[2][b], orig_predicted_detection_maps[3][b],
                                        orig_predicted_detection_maps[4][b]])

                            elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                                count_sample['points_annot'].append([[current_count]])



                if len(crops_count_GT)==0:
                    count_sample = None
                else:
                    if config.Counting.counting_type == 'withKeyPoints':
                        all_crops_GT_detections_maps = torch.tensor(all_crops_GT_detections_maps)
                        if len(all_predicted_detection_maps)>1:
                            for i in range(len(all_predicted_detection_maps)):
                                all_predicted_detection_maps[i] = all_predicted_detection_maps[i].unsqueeze(dim=0)

                            all_predicted_detection_maps = torch.cat(all_predicted_detection_maps, dim=0)
                        else:
                            all_predicted_detection_maps = all_predicted_detection_maps[0].unsqueeze(dim=0)

            if count_sample is None:
                # relevant
                if verbose:
                    printf(
                        "##############################################################################################\n")
                    printf("image: %s\n", image_name)
                    printf(
                        "##############################################################################################\n")
                    printf("No gt points in any crop...\n")
                    print()

                continue



            ############################################################################################################
            # Preparing the evaluation in comparison to the gt points' count of the relevant gt object -
            # the object with iou>thresh of the predicted box with the gt box
            ############################################################################################################

            if to_draw:
                gt_boxes = data['bbox_annot'][0]
                for i in range(gt_boxes.shape[0]):
                    x1 = gt_boxes[i, 0].numpy()/scale
                    y1 = gt_boxes[i, 1].numpy()/scale
                    x2 = gt_boxes[i, 2].numpy()/scale
                    y2 = gt_boxes[i, 3].numpy()/scale
                    draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)

                #orig_img.show(title=image_name)
                # save the original img with pred bbox and gt anns
                orig_img.save(os.path.join(draw_path,image_name))

            detected_annotations = []
            scores_temp = np.zeros((0,))

            for i in range(len(adjusted_crops_orig_boxes)):
                scores_temp = np.append(scores_temp, adjusted_crops_orig_boxes[i][4])
                adjusted_crops_orig_boxes[i] = torch.tensor(adjusted_crops_orig_boxes[i]).unsqueeze(dim=0)

            adjusted_crops_orig_boxes = torch.cat(adjusted_crops_orig_boxes, dim=0)
            indices = np.argsort(-scores_temp)
            adjusted_crops_orig_boxes = adjusted_crops_orig_boxes[indices]

            for i in range(box_annotations.shape[0]):
                box_annotations[i, 0] = box_annotations[i, 0] * scale[0]
                box_annotations[i, 1] = box_annotations[i, 1] * scale[0]
                box_annotations[i, 2] = box_annotations[i, 2] * scale[0]
                box_annotations[i, 3] = box_annotations[i, 3] * scale[0]

            max_overlap_array = []
            for i in range(box_annotations.shape[0]):
                box_annotations[i, 0] = box_annotations[i, 0] * scale[0]
                box_annotations[i, 1] = box_annotations[i, 1] * scale[0]
                box_annotations[i, 2] = box_annotations[i, 2] * scale[0]
                box_annotations[i, 3] = box_annotations[i, 3] * scale[0]

            for d in adjusted_crops_orig_boxes:
                # if box_annotations.shape[0] == 0:
                #     continue
                # from orig coordinates to the resized - to be on the same scale like adjusted_crops_orig_boxes
                # if (not self.training) and config.detect_and_count.choose_by_IoUandPrc_Flag:
                #     print('img:', img_info['name'])
                #
                #     bbox_pred = self.choose_boxes_by_IoUandPrc(bbox_pred, detection_anns[:,:,:5], box_scores.clone().detach())

                overlaps = compute_overlap(np.expand_dims(d, axis=0), np.array(box_annotations))
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= config.Detection.iou_threshold and assigned_annotation not in detected_annotations:
                    detected_annotations.append(assigned_annotation)
                    max_overlap_array.append(max_overlap)
                    found_orig_objects += 1

                    gt_box_id = box_annotations[assigned_annotation[0]][5]
                    # find the gt count value of the assigned gt box
                    has_points=False
                    for g in gt_counts:
                        if g[2] == gt_box_id:
                            gt_count = g[0]
                            orig_count_GT.append(gt_count)
                            has_points=True
                            break
                    # matched a gt box without gt points
                    if not has_points:
                        orig_count_GT.append(-1)
                        matched_without_gt_points+=1
                else:
                    orig_count_GT.append(-1)
                    # max_overlap_array.append(-1)
                    max_overlap_array.append(max_overlap)
                    FP += 1

            ############################################################################################################
            # Evaluate the results of the given image
            ############################################################################################################

            if len(count_outputs):
                if config.Counting.counting_type == 'withKeyPoints':
                    if config.Counting.calc_det_performance:
                        for b in range(all_predicted_detection_maps.shape[0]):
                            t, p = detection_evaluation(all_predicted_detection_maps[b, :, :], all_crops_GT_detections_maps[b, :, :])
                            T = T + t
                            P = P + p

                all_crops_GT_counts.append(crops_count_GT)
                all_predicted_counts.append(np.round(count_pred))
                all_orig_GT_counts.append(orig_count_GT)

                if verbose:
                    printf("#######################################################################################\n")
                    printf("image: %s\n", image_name)
                    printf("#######################################################################################\n")

                for i in range(len(crops_count_GT)):

                    crops_abs_diff.append(abs(crops_count_GT[i] - count_pred[i]))
                    crops_rel_error.append(abs(crops_count_GT[i] - np.round(count_pred[i])) / crops_count_GT[i])
                    if orig_count_GT[i] != -1:
                        orig_abs_diff.append(abs(orig_count_GT[i] - count_pred[i]))
                        orig_rel_error.append(abs(orig_count_GT[i] - np.round(count_pred[i])) / orig_count_GT[i])

                    if verbose:
                        printf(
                            "box: %d | crops_count_GT: %d | count_predicted: %d (%.3f) | abs_diff: %.3f | rel_error: %.3f\n",
                            i, np.round(crops_count_GT[i]), np.round(count_pred[i]), count_pred[i],
                            crops_abs_diff[-1], crops_rel_error[-1])
                        if orig_count_GT[i] != -1:
                            printf(
                                "orig_count_GT: %d | orig_abs_diff: %.3f | orig_rel_error: %.3f\n",
                                int(orig_count_GT[i]), orig_abs_diff[-1], orig_rel_error[-1])
                            print('max_overlap = ', max_overlap_array[i])
                        else:
                            print('No gt box with iou>=', config.Detection.iou_threshold)
                            print('max_overlap = ', max_overlap_array[i])
                        print()


                    if to_draw:

                        bbox_crop = count_sample['img'][i].clone()

                        # view the crops per image

                        bbox_img = bbox_crop.cpu().clone().detach()
                        bbox_img = np.array(255 * unnormalize(bbox_img))
                        bbox_img[bbox_img < 0] = 0
                        bbox_img[bbox_img > 255] = 255
                        bbox_img = np.transpose(bbox_img, (1, 2, 0))

                        img2 = transforms.ToPILImage(mode='RGB')(bbox_img.astype(np.uint8))
                        imgToVis = img2.copy()

                        draw2 = PIL.ImageDraw.Draw(img2)
                        for j in range(len(relevant_points[i])): #points per box
                            x = relevant_points[i][j]['x']
                            y = relevant_points[i][j]['y']
                            draw2.ellipse(((int(x)-10, int(y)-10), (int(x) + 10, int(y) + 10)), fill="black", width=config.DrawProperties.LINE_WIDTH)

                        #img2.show()
                        box_name = image_name + '_crop_' + str(i) + '.jpg'  # image_name.split('.jpg')[0]+'_crop_'+str(i)+'.jpg'
                        img2.save(os.path.join(draw_path, box_name))

                        if config.Counting.counting_type == 'withKeyPoints':
                            true_maps = []
                            for p in [1, 2, 3, 4, 5]:
                                true_maps.append(count_sample['points_annot'][i][p][0].copy())

                            pred_maps = []
                            for p in range(5):
                                pred_maps.append(all_predicted_detection_maps_toDraw[i][p])

                            # visualize all maps
                            # for p in range(5):
                            #     visualize_images(pred_maps[p], true_maps[p],
                            #                        image_name+'_crop_'+str(i)+'_map_'+str(p+1), imgToVis, draw_path)

                            # Visualize the fifth map
                            if config.DrawProperties.DRAW_MAPS:
                                visualize_images(pred_maps[4], true_maps[4],
                                                 image_name + '_crop_' + str(i) + '_map_' + str(4 + 1), imgToVis, draw_path)

                            # draw for model 2:
                            # img2.show()
                            # visualize_images(count_outputs[1][i], count_sample['points_annot'][1][i],
                            #                  image_name.split('.jpg')[0] + '_' + str(i) + '_' + str(1), dataset,
                            #                  model, imgToVis)
                            # for p in range(1,6):
                            #     visualize_images(count_outputs[p][i], count_sample['points_annot'][p][i],
                            #                      image_name.split('.jpg')[0]+'_'+str(i)+'_'+str(p), dataset, model, imgToVis)



        ################################################################################################################
        # Get results summary
        ################################################################################################################

        # print('Get results summary:')

        if len(all_predicted_counts)==0:
            if verbose:
                print('There are no images with predicted boxes')
            return []


        num_of_images = len(all_crops_GT_counts)
        # gather all results
        total_crops_GT_counts = []
        total_predicted_counts = []

        total_orig_GT_counts = []
        total_predicted_for_orig_boxes = []
        total_orig_box_for_count=0

        for n in range(num_of_images):
            for j in range(len(all_crops_GT_counts[n])):
                total_crops_GT_counts.append(all_crops_GT_counts[n][j])
                total_predicted_counts.append(all_predicted_counts[n][j])

                if all_orig_GT_counts[n][j] != -1:
                    total_orig_GT_counts.append(all_orig_GT_counts[n][j])
                    total_orig_box_for_count+=1
                    total_predicted_for_orig_boxes.append(all_predicted_counts[n][j])

        total_crop_boxes = len(total_crops_GT_counts)

        crops_avg_abs_count_diff = SumOfAbsDifferences(total_crops_GT_counts, total_predicted_counts) / total_crop_boxes
        crops_avg_rel_error = np.mean(crops_rel_error)
        crops_MSE = np.mean((np.array(total_crops_GT_counts) - np.array(total_predicted_counts)) ** 2)

        if total_orig_box_for_count>0:
            orig_avg_abs_count_diff = SumOfAbsDifferences(total_orig_GT_counts, total_predicted_for_orig_boxes) / total_orig_box_for_count
            orig_avg_rel_error = np.mean(orig_rel_error)
            orig_MSE = np.mean((np.array(total_orig_GT_counts) - np.array(total_predicted_for_orig_boxes)) ** 2)
        else:
            print('No gt boxes for any image')
            return []

        crops_count_agr = 0
        for i in range(total_crop_boxes):
            if total_crops_GT_counts[i] == total_predicted_counts[i]:
                crops_count_agr += 1
        crops_count_agr = crops_count_agr / total_crop_boxes

        orig_count_agr = 0
        for i in range(total_orig_box_for_count):
            if total_orig_GT_counts[i] == total_predicted_for_orig_boxes[i]:
                orig_count_agr += 1
        orig_count_agr = orig_count_agr / total_orig_box_for_count

        crops_var_GT_counts = np.var(total_crops_GT_counts)
        crops_FVU = crops_MSE/crops_var_GT_counts

        orig_var_GT_counts = np.var(total_orig_GT_counts)
        orig_FVU = orig_MSE / orig_var_GT_counts
        orig_mean_GT_counts = np.mean(total_orig_GT_counts)

        precision_det = found_orig_objects/(found_orig_objects+FP)
        if verbose:

            printf("====================================================================================================\n")
            printf("Summary - for crops count \n")

            printf("crops_avg_abs_count_diff: %.3f | crops_count_agreement: %.3f | crops_MSE: %.3f | crops_avg_relative_error: %.3f | crops_1-FVU: %3f\n",
                crops_avg_abs_count_diff, crops_count_agr, crops_MSE, crops_avg_rel_error, 1-crops_FVU)
            print()

            printf("====================================================================================================\n")
            printf("Summary - for gt boxes count \n")
            printf(
                "orig_avg_abs_count_diff: %.3f | orig_count_agreement: %.3f | orig_MSE: %.3f | orig_avg_relative_error: %.3f | orig_1-FVU: %3f\n",
                orig_avg_abs_count_diff, orig_count_agr, orig_MSE, orig_avg_rel_error, 1 - orig_FVU)
            print()

            print(
                "====================================================================================================\n")
            printf("General Stats\n")
            printf("gt_objects_withGTpoints = %d\n", gt_objects_withGTpoints)
            printf("found_orig_objects = %d (%.2f%% of gt objects [recall])\n", found_orig_objects,
                   100 * found_orig_objects / gt_objects_withGTpoints)
            printf("precision = %.3f\n\n", precision_det)


            printf("====================================================================================================\n")
            printf("GT data summary\n")
            printf("all_orig_avg_GT_counts: %.3f\n",   np.mean(all_data_gt_count))
            printf("all_orig_var_GT_counts: %.3f\n",   np.var(all_data_gt_count))
            printf("all_orig_std_GT_counts: %.3f\n\n", np.sqrt(np.var(all_data_gt_count)))

            printf("====================================================================================================\n")
            printf("Found GT data summary\n")
            printf("orig_avg_GT_counts: %.3f\n",   orig_mean_GT_counts)
            printf("orig_var_GT_counts: %.3f\n",   orig_var_GT_counts)
            printf("orig_std_GT_counts: %.3f\n\n", np.sqrt(orig_var_GT_counts))

            print("=====================================================================================================\n")
            print('Crops stats\n')
            print('num of crops = ', len(predicted_counts_any_crop))
            print('FP = ', FP)
            print('found_orig_objects = ', found_orig_objects)
            print('matched_without_gt_points = ', matched_without_gt_points)
            print('crops_without_gt_points = ', crops_without_gt_points)
            print('summaries to num of crops ?')
            print('avg pred count per crop = ', np.mean(predicted_counts_any_crop))
            print('var of pred count per crop = ', np.var(predicted_counts_any_crop))

            print("=====================================================================================================\n")
            print('Per image gt stats\n')
            print('avg of image averages = ', np.mean(per_im_avg))
            print('var of image averages = ', np.var(per_im_avg))
            print("per image avg: ", np.round(per_im_avg, 2))




        model.train()

        if config.Counting.calc_det_performance:
            recall, precision, ap = calc_recall_precision_ap(T, P)
            plot_RP_curve(recall, precision, ap, save_path=config.General.experiment_path)

        out = [orig_avg_rel_error,
               gt_objects_withGTpoints,
               found_orig_objects,
               found_orig_objects/gt_objects_withGTpoints,
               precision_det,
               orig_avg_abs_count_diff,
               orig_count_agr,
               orig_MSE,
               (1 - orig_FVU)]

        return out









