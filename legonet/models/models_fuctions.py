import torch

import torchvision.ops
import numpy as np
import cv2

import config
from legonet.myDataloader import UnNormalizer


##########################################################################################################

def get_detection_output(self, transformed_anchors, classification_vector):

    scores = torch.max(classification_vector, dim=2, keepdim=True)[0]

    scores_over_thresh = (scores > self.min_score)[0, :, 0]

    if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just return
        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

    classification = classification_vector[:, scores_over_thresh, :]
    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
    scores = scores[:, scores_over_thresh, :]

    anchors_nms_idx = torchvision.ops.nms(transformed_anchors[0, :, :], scores[0, :, 0], config.Detection.NMS_THRESHOLD)

    nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

    return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


def get_crops(self, img, bbox_pred=None, anns=None, view_gt = False):
    if view_gt:
        unnormalize = UnNormalizer()

        im = img.cpu().clone().detach()
        im = np.array(255 * unnormalize(im))
        im[im < 0] = 0
        im[im > 255] = 255
        im = np.transpose(im, (1 ,2 ,0))
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imshow('img', im)
        cv2.waitKey(0)

        # draw gt boxes for each img - based on anns args.do_counting
        bbox_tensor = anns.cpu()
        bbox = bbox_tensor.clone()
        x1 = bbox[:, 0].int()
        y1 = bbox[:, 1].int()
        x2 = bbox[:, 2].int()
        y2 = bbox[:, 3].int()

        for j in range(x1.shape[0]):
            cv2.rectangle(im, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])),
                          color=(0, 0, 255), thickness=2)
        cv2.imshow('img' ,im)
        cv2.waitKey(0)

        im = torch.tensor(im).float().permute(2,0,1).unsqueeze(dim=0).to(config.General.device)

    indices = torch.tensor([0, 1, 2, 3]).cuda()
    box_coord = torch.index_select(bbox_pred, 1, indices)

    bbox_crops = self.roi_align(input=img.unsqueeze(dim=0), boxes=[box_coord],
                                output_size=(config.Counting.crops_size[0], config.Counting.crops_size[1]))

    # view the crops per image
    if view_gt:
        bbox_img = np.asarray(bbox_crops[0].permute(1 ,2 ,0).cpu())
        bbox_img = bbox_img.astype(np.uint8)
        cv2.imshow('bbox', bbox_img)
        cv2.waitKey(0)

    return bbox_crops

def image_output_shape(self, image_shape, pyramid_level=3):
    return (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)


def images_ratios(self, image_shape, output_shape):
    return output_shape / np.array(image_shape[:2])


def create_gausian_mask(self, center_point, nCols, nRows, q=99, radius=(5, 5)):
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

    x2 = (((x - round(center_point[0])) * s) / radius[0]) ** 2
    y2 = (((y - round(center_point[1])) * s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return p


def compute_keypoints_targets_multi_maps(self, image_shape, annotations_points_centers_a, radius=(5, 5), pyramid_level=3):
    # resize transformed-image and annotations
    import copy
    annotations_points_centers = copy.deepcopy(annotations_points_centers_a)
    # here we should resize image too and then check it with the annotations
    output_shape = self.image_output_shape(image_shape, pyramid_level=pyramid_level)
    image_ratio = self.images_ratios(image_shape, output_shape)

    if len(annotations_points_centers) == 0:
        return [np.zeros(output_shape)]

    annotations = np.zeros(output_shape)
    for i in range(len(annotations_points_centers)):
        annotations_points_centers[i]['y'] *= image_ratio[0]
        annotations_points_centers[i]['x'] *= image_ratio[1]

        current_points = [annotations_points_centers[i]['x'], annotations_points_centers[i]['y'] ]
        gaussian_map = self.create_gausian_mask(current_points[:2], output_shape[1], output_shape[0],
                                                radius=radius)
        # each center point in the GT will be 1 in the annotation map
        annotations = np.maximum(annotations, gaussian_map)

        if np.isnan(annotations).any():
            raise ("nan was found")

    return annotations


def getitem(self, bbox_crops, points = None, anns = None):
    filtered_samples = []
    for b in range(len(bbox_crops)):
        if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
            if self.training or (not self.training and anns is not None):
                current_img = bbox_crops[b][0][0].permute(1, 2, 0).cpu()
            else:
                current_img = bbox_crops[b][0].permute(1, 2, 0).cpu()
        else:
            current_img = bbox_crops[b][0].permute(1 ,2 ,0).cpu()
        sample = {'img': current_img}

        # sample['lean_version'] = self.lean_version

        if points:
            current_ann = points[b]

            if len(current_ann) > 0:
                annotations_group_num_of_points = len(current_ann)
                annotations_group_points_center = current_ann
                annotation_map_1 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                             annotations_group_points_center,
                                                                             radius=config.Counting.map_1_R) # (5,7))
                annotation_map_2 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                             annotations_group_points_center,
                                                                             radius=config.Counting.map_2_R) # (3, 5))
                annotation_map_3 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                             annotations_group_points_center,
                                                                             radius= config.Counting.map_3_R) # (3, 7))
                annotation_map_4 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                             annotations_group_points_center,
                                                                             radius=config.Counting.map_4_R) # (5, 5))
                annotation_map_5 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                             annotations_group_points_center,
                                                                             radius=config.Counting.map_5_R) # (3,3)

                # plt.imsave('path' + '/' + 'ann1' + '_gt.png', annotation_map_1)
                # plt.imsave('path' + '/' + 'ann2' + '_gt.png', annotation_map_2)
                # plt.imsave('path' + '/' + 'ann3' + '_gt.png', annotation_map_3)
                # plt.imsave('path' + '/' + 'ann4' + '_gt.png', annotation_map_4)
                # plt.imsave('path' + '/' + 'ann5' + '_gt.png', annotation_map_5)

                sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                   annotation_map_3, annotation_map_4, annotation_map_5]

            elif self.network_type == "both_for_roots_2":
                annotations_group_num_of_points = 0
                annotation_map_1 = torch.empty((80 ,80), dtype=float)
                annotation_map_2 = torch.empty((80, 80), dtype=float)
                annotation_map_3 = torch.empty((80, 80), dtype=float)
                annotation_map_4 = torch.empty((80, 80), dtype=float)
                annotation_map_5 = torch.empty((80, 80), dtype=float)

                sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                   annotation_map_3, annotation_map_4, annotation_map_5]


            if self.network_type =="both_for_roots_2":

                anns_idx = bbox_crops[b][1].item()

                if anns_idx !=-1:
                    anns_loc = ((anns[1][0][0][: ,2] == anns_idx).nonzero(as_tuple=True)[0]).item()
                    roots_anns = anns[1][0][0][anns_loc]  # [int(anns_idx)]

                    points_color = roots_anns[0].unsqueeze(dim=0) # points_count
                    length = roots_anns[3].unsqueeze(dim=0)
                    diameter = roots_anns[4].unsqueeze(dim=0)
                    gt_idx = torch.tensor(anns_idx, dtype=float).unsqueeze(dim=0)
                    sample['roots_annot'] = torch.cat([points_color, length, diameter, gt_idx]) # points_count

                else:
                    sample['roots_annot'] = torch.tensor([-1, 0, 0, -1], dtype=float) # [color, length, dia, gt_box_id]

                if config.detect_with_points.detect_points:
                    sample['gt_box_id' ] =anns_idx
                    sample['gt_box_maps'] = []
                    for i in range(5):
                        sample['gt_box_maps'].append(np.array(anns[2][1][anns_loc][i]))  #


        filtered_samples.append(sample)

    return filtered_samples


def find_points_in_bbox(self, img, point_anns, bbox_pred, scale, network_type):

    unnormalize = UnNormalizer()

    points =[]

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
        box_x1, box_y1, box_x2, box_y2, box_id = bbox_pred[b, :5]  # box_x1, box_y1, box_x2, box_y2 = bbox_pred[b,:4]
        cv2.rectangle(im, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), color=(0, 0, 255), thickness=2)

        for p in point_anns:
            if network_type == "both_for_roots" or network_type == "both_for_roots_2":
                if p['bbox_id'] != box_id:
                    continue
            if p['x'] <= box_x2 and p['x'] >= box_x1 and p['y'] <= box_y2 and p['y'] >= box_y1:
                p_x.append(p['x'])
                p_y.append(p['y'])

        points.append({'x': p_x, 'y': p_y})

    # cv2.imshow('img', im)
    # cv2.waitKey(0)

    return points


def nmcs(self, predicted_boxes, relevant_points):
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
