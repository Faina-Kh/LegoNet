import torch
import torch.nn as nn
import torchvision.ops
from torchvision.ops import roi_align
import copy
import cv2
import numpy as np
from itertools import compress

from legonet import legos
from legonet import utils
import config
import legonet.my_dataloader as myDataloader
from legonet.my_dataloader import UnNormalizer
from legonet.eval.both_eval_new_241 import choose_boxes_by_IoUandPrc

from model_bbox_detection import BBOX_Detection





class PerObjectEstimate(nn.Module):

    def __init__(self, dataset, network_type, num_classes, freeze_detection = True):
        super(PerObjectEstimate, self).__init__()

        self.dataset = dataset
        self.network_type = network_type
        self.num_classes = num_classes
        self.freeze_detection = freeze_detection

        self.bbox_detection = BBOX_Detection(num_classes = num_classes, freeze_detection = self.freeze_detection)
        self.backbone_2 = legos.ResNetBackboneModule(name='backbone_for_attribute') #name='backbone_for_count'

        self.find_2 = legos.FindModule(num_classes=num_classes, name='find_for_attribute', task='attribute_estimation',
                                         Find_for_count = True) #name='find_for_count', task='counting'

        if config.AttributeEstimation.estimate_type == 'withKeyPoints': #'reg_fpn_p3_p7_min_sig'
            self.estimator = legos.KeypointBasedEstimator() #(num_classes, inter_losses = config.Counting.inter_losses)

        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
            self.estimator = legos.RegressionBasedEstimator(num_classes)

        self.roi_align = roi_align

        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_detector(self):
        self.bbox_detection.eval()
        # for p in self.bbox_detection.parameters():
        #     p.requires_grad = False

        # freeze gradients of detection
        for param in self.bbox_detection.backbone_1.parameters():
            param.requires_grad = False

        for param in self.bbox_detection.find_1.parameters():
            param.requires_grad = False

        for param in self.bbox_detection.where.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        img_batch, annotations, group_idx = inputs

        if annotations is None:
            detection_anns, counting_anns = None, None
        else:
            detection_anns, counting_anns = annotations

        bbox_crops_list = []
        relevant_points_anns = []
        crops_orig_boxes = []

        estimation_outputs = None #counting_outputs

        for img_idx in range(img_batch.shape[0]):
            if not self.bbox_detection.training: #self.freeze_detection:
                detection_outputs = self.bbox_detection([img_batch[img_idx].unsqueeze(dim=0)])
            else: #elif self.training:
                classification_loss, regression_loss = self.bbox_detection(img_batch[img_idx].unsqueeze(dim=0))

            # detection eval output:
            if (detection_outputs[0].to(config.General.device)).equal(torch.zeros(0).to(config.General.device)):
                bbox_pred_adjusted=torch.empty(0)
                continue

            else:
                img_id = self.dataset.image_ids[group_idx[img_idx]]
                img_info = self.dataset.img_info[img_id]

                # continue with attribute estimation
                bbox_pred = detection_outputs[2].detach() #b,n,x1,y1,x2,y2 grad=true
                box_scores = detection_outputs[0]

                if counting_anns is not None:
                    # find the gt box for each detection
                    pred_to_box_idx = choose_boxes_by_IoUandPrc(bbox_pred, detection_anns, box_scores)

                if (self.training or (
                        not self.training and annotations is not None)) and counting_anns is not None:  # and config.detect_with_points.detect_points: #

                    # for box in pred_to_box_idx:
                    #     # box ids start with 1, not 0
                    #     if box[:, 4] != -1:
                    #         box[:, 4] = box[:, 4] + 1

                    bbox_pred = torch.cat(pred_to_box_idx, dim=0)

                    if self.training:
                        # keep only the annotations of detected boxes
                        box_scores = box_scores[bbox_pred[:, 4] != -1]
                        bbox_pred = bbox_pred[bbox_pred[:, 4] != -1]

                    true_ids = bbox_pred[:, 4].sort().values

                    if not self.training:
                        true_ids = true_ids.cpu()

                    # img_batch, annotations, group_idx, do_counting = inputs
                    # detection_anns, counting_anns, per_obj_maps = annotations

                    anns_box_ids = detection_anns[0][:, 5]
                    ans_detected_ids = []
                    for i in range(anns_box_ids.shape[0]):
                        if anns_box_ids[i].item() != -1: # check if it's a box with points
                            ans_detected_ids.append(anns_box_ids[i] in true_ids)

                    detection_anns = detection_anns[0][ans_detected_ids].unsqueeze(dim=0)
                    if len(ans_detected_ids)>0:
                        counting_anns[0][0] = counting_anns[0][0][ans_detected_ids]

                        anns_box_ids_p = counting_anns[0][1][:, 3].float()
                        ans_detected_ids_p = []
                        for i in range(counting_anns[0][1].shape[0]):
                            ans_detected_ids_p.append(anns_box_ids_p[i] in true_ids.cpu())

                        counting_anns[0][1] = counting_anns[0][1][ans_detected_ids_p]

                    annotations = [detection_anns, counting_anns]

                ##########################################################################
                bbox_pred = torch.cat((bbox_pred, box_scores.unsqueeze(-1)), dim=-1) #[x1,y1,x2,y2,gt_box_id,score]

                if len(bbox_pred)==0:
                    continue

                bbox_pred_adjusted = bbox_pred.clone()

                # Get gt points' annotations for training for the KeyPoints-based model
                if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    points=None
                    if self.training or (not self.training and annotations is not None):
                        point_anns = self.dataset.image_data_points_location[img_info['name']]
                        point_anns_copy = []
                        for d in point_anns:
                            point_anns_copy.append(copy.deepcopy(d))
                        points = self.find_points_in_bbox(img_batch[img_idx], point_anns_copy, bbox_pred_adjusted, img_info['scale'], self.network_type)
    
                        if config.AttributeEstimation.do_nmcs:# don't do for roots
                            non_suppressed_indices = self.nmcs(bbox_pred_adjusted, points)
                            points = list(compress(points, non_suppressed_indices))
                            bbox_pred_adjusted = bbox_pred_adjusted[non_suppressed_indices, :]


                # get the predicted crops
                #    in inference - keep all predicted boxes
                #    in training - keep only those that include points

                if counting_anns is not None or (not self.training and config.General.predict_empty_image):

                    #empty_crops_gtidx = []

                    for b in range(bbox_pred_adjusted.shape[0]):

                        x1 = bbox_pred_adjusted[b, 0]
                        y1 = bbox_pred_adjusted[b, 1]
                        x2 = bbox_pred_adjusted[b, 2]
                        y2 = bbox_pred_adjusted[b, 3]

                        current_score = float(bbox_pred_adjusted[b, -1].cpu())

                        #box_idx = bbox_pred_adjusted[b, 4].item()

                        crops_orig_boxes.append([float(x1.cpu()), float(y1.cpu()), float(x2.cpu()), float(y2.cpu()), current_score])

                        if detection_anns is None:
                            bbox_crops = self.get_crops(img_batch[img_idx],
                                                        bbox_pred=bbox_pred_adjusted[b].unsqueeze(dim=0),
                                                        anns=None)
                        else:
                            bbox_crops = self.get_crops(img_batch[img_idx],
                                                        bbox_pred=bbox_pred_adjusted[b].unsqueeze(dim=0),
                                                        anns=detection_anns[img_idx])


                        if self.training or (not self.training and annotations is not None): # training

                            if points is not None:
                                current_points = points[b]
                                points_to_view = []

                                # for boxes that include points
                                if len(current_points['x'])>0:
                                    current_points['x'] = current_points['x'] - (x1.cpu().numpy()) * np.ones(len(current_points['x']))
                                    current_points['y'] = current_points['y'] - (y1.cpu().numpy()) * np.ones(len(current_points['y']))

                                    scale_x = config.AttributeEstimation.crops_size[0] / (x2 - x1).cpu().numpy()
                                    scale_y = config.AttributeEstimation.crops_size[1] / (y2 - y1).cpu().numpy()
                                    current_points['x'] = current_points['x']*scale_x
                                    current_points['y'] = current_points['y']*scale_y

                                    for i in range(len(current_points['x'])):
                                       points_to_view.append({'x':current_points['x'][i], 'y':current_points['y'][i]})
                                    #self.view_points_on_img(bbox_crops, points_to_view)

                                    relevant_points_anns.append(points_to_view)
                                    bbox_crops_list.append(bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).to(config.General.device))

                                else:
                                    if not self.training:
                                        bbox_crops_list.append(bbox_crops)
                                        relevant_points_anns.append([])

                            else:
                                bbox_crops_list.append(bbox_crops)

                        else:
                            bbox_crops_list.append(bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).to(config.General.device))

        including_counting = False

        if len(bbox_crops_list)>0:

            num_of_boxes = bbox_pred_adjusted.shape[0]

            if self.training:
                sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points_anns)
            else:
                if annotations is not None and points is not None:
                    sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points_anns,
                                               anns=annotations)
                else:
                    sample_list = self.getitem(bbox_crops=bbox_crops_list)

            sample_anns = myDataloader.kcsv_collater_2(sample_list)

            if self.training or (not self.training and annotations is not None and points is not None):
                corrected_counting_anns = sample_anns['points_annot']
                corrected_counting_anns = [a.to(config.General.device) for a in corrected_counting_anns]

            num_of_crops = sample_anns['img'].shape[0]

            # img input should be tensor [b,c,h,w]
            bbox_pyramid_feats= []
            bbox_pyramid_p3 = []

            for i in range(num_of_crops):
                current= self.backbone_2(sample_anns['img'][i].unsqueeze(dim=0).to(config.General.device))
                bbox_pyramid_feats.append(current)
                if i==0:
                    bbox_pyramid_p3.append(bbox_pyramid_feats[i][0]) ##current[0]
                else:
                    bbox_pyramid_p3[0] = torch.cat((bbox_pyramid_p3[0], bbox_pyramid_feats[i][0]), dim=0)  #[bbox_pyramid_feats[0]]  # [bbox_pyramid_feats]  # [p3]


            if self.training:

                if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    count_train_inputs = bbox_pyramid_p3, corrected_counting_anns[1:6]  # anchors, None

                    l1 = 0
                    maps_loss = 0
                    for i in range(num_of_boxes):
                        current_count_train_inputs = [[count_train_inputs[0][0][i].unsqueeze(dim=0)],
                                                      [count_train_inputs[1][0][i].unsqueeze(dim=0), count_train_inputs[1][1][i].unsqueeze(dim=0),
                                                       count_train_inputs[1][2][i].unsqueeze(dim=0), count_train_inputs[1][3][i].unsqueeze(dim=0),
                                                       count_train_inputs[1][4][i].unsqueeze(dim=0)]]

                        SFMS_lists, cls_output, current_maps_loss = self.find_2(current_count_train_inputs)
                        maps_loss += current_maps_loss

                        count_input = SFMS_lists, cls_output, corrected_counting_anns[0]  # annotations[1][0][0] #corrected_counting_anns
                        current_l1 = self.estimator(count_input)[0]
                        l1 += current_l1

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    counting_loss = self.estimator([bbox_pyramid_feats[:config.AttributeEstimation.num_of_pyr_levels], corrected_counting_anns[0]])

                including_counting = True

            else:
                if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                    SFMS_lists, cls_output = self.find_2(bbox_pyramid_p3)
                    count_input = SFMS_lists, cls_output

                    estimation_outputs = self.estimator(count_input)

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    estimation_outputs = self.estimator(bbox_pyramid_feats[:config.Counting.num_of_pyr_levels])
                    
        else:
            sample_anns=None

        if self.training:
            if not self.bbox_detection.training:
                classification_loss = None
                regression_loss = None

            if counting_anns is not None:
                if not including_counting: # don't have bbox predictions
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                        return classification_loss, regression_loss, None, None

                    elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                        return classification_loss, regression_loss, None


                if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    return classification_loss, regression_loss, l1, maps_loss  #counting_loss #total_loss #classification_loss, regression_loss, counting_loss

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    return classification_loss, regression_loss, counting_loss


        else:
            if bbox_pred_adjusted.shape[0] > 0: #len(bbox_crops_list)>0:
                return detection_outputs, estimation_outputs, sample_anns, relevant_points_anns, crops_orig_boxes #bbox_pred_adjusted,....

            else:
                return detection_outputs, estimation_outputs, None, None, None  #[],...

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
            im = np.transpose(im, (1,2,0))
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imshow('img', im)
            cv2.waitKey(0)

            #draw gt boxes for each img - based on anns args.do_counting
            bbox_tensor = anns.cpu()
            bbox = bbox_tensor.clone()
            x1 = bbox[:, 0].int()
            y1 = bbox[:, 1].int()
            x2 = bbox[:, 2].int()
            y2 = bbox[:, 3].int()

            for j in range(x1.shape[0]):
                cv2.rectangle(im, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])),
                              color=(0, 0, 255), thickness=2)
            cv2.imshow('img',im)
            cv2.waitKey(0)

            im=torch.tensor(im).float().permute(2,0,1).unsqueeze(dim=0).to(config.General.device)

        indices = torch.tensor([0, 1, 2, 3]).to(config.General.device)
        box_coord = torch.index_select(bbox_pred, 1, indices)

        bbox_crops = self.roi_align(input=img.unsqueeze(dim=0), boxes=[box_coord],
                                    output_size=(config.AttributeEstimation.crops_size[0], config.AttributeEstimation.crops_size[1]))
        #self.roi_align(input=img.unsqueeze(dim=0), boxes=[bbox_pred], output_size=(config.Counting.crops_size[0], config.Counting.crops_size[1]))
        # check if:
        # input (Tensor[N, C, H, W])
        # boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – the box coordinates in (x1, y1, x2, y2)
        # output_size – the size of the output after the cropping is performed, as (height, width)
        # output(Tensor[K, C, output_size[0], output_size[1]])

        # view the crops per image
        if view_gt:
            bbox_img = np.asarray(bbox_crops[0].permute(1,2,0).cpu())
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
            current_img = bbox_crops[b][0].permute(1,2,0).cpu()
            sample = {'img': current_img}
            #sample['lean_version'] = self.lean_version

            if points:
                current_ann = points[b]

                if len(current_ann) > 0:
                    annotations_group_num_of_points = len(current_ann)
                    annotations_group_points_center = current_ann
                    annotation_map_1 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.AttributeEstimation.map_1_R) #(5,7))
                    annotation_map_2 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.AttributeEstimation.map_2_R) #(3, 5))
                    annotation_map_3 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius= config.AttributeEstimation.map_3_R) #(3, 7))
                    annotation_map_4 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.AttributeEstimation.map_4_R) #(5, 5))
                    annotation_map_5 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.AttributeEstimation.map_5_R) #(3,3)

                    # plt.imsave('path' + '/' + 'ann1' + '_gt.png', annotation_map_1)
                    # plt.imsave('path' + '/' + 'ann2' + '_gt.png', annotation_map_2)
                    # plt.imsave('path' + '/' + 'ann3' + '_gt.png', annotation_map_3)
                    # plt.imsave('path' + '/' + 'ann4' + '_gt.png', annotation_map_4)
                    # plt.imsave('path' + '/' + 'ann5' + '_gt.png', annotation_map_5)

                    sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                                       annotation_map_3, annotation_map_4, annotation_map_5]

                elif not self.training:
                    annotations_group_num_of_points = 0
                    annotation_map_1 = torch.empty((80,80), dtype=float)
                    annotation_map_2 = torch.empty((80, 80), dtype=float)
                    annotation_map_3 = torch.empty((80, 80), dtype=float)
                    annotation_map_4 = torch.empty((80, 80), dtype=float)
                    annotation_map_5 = torch.empty((80, 80), dtype=float)

                    sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                       annotation_map_3, annotation_map_4, annotation_map_5]

            filtered_samples.append(sample)

        return filtered_samples

    def find_points_in_bbox(self, img, point_anns, bbox_pred, scale, network_type):

        unnormalize = UnNormalizer()

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
            box_x1, box_y1, box_x2, box_y2, box_id = bbox_pred[b,:5] #box_x1, box_y1, box_x2, box_y2 = bbox_pred[b,:4]
            cv2.rectangle(im, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), color=(0, 0, 255), thickness=2)

            for p in point_anns:
                if p['x']<=box_x2 and p['x']>=box_x1 and p['y']<=box_y2 and p['y']>=box_y1:
                    p_x.append(p['x'])
                    p_y.append(p['y'])

            points.append({'x':p_x, 'y':p_y})

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

    def view_points_on_img(self, img, point_anns):

        for p in point_anns:
            cv2.circle(img, (int(p['x']), int(p['y'])), radius=3, color=(0, 0, 255), thickness=2)

        cv2.imshow('img', img)
        cv2.waitKey(0)



def main():

    m = torch.load("D:\\PyCharmProjects\\LEGONet\\lean_weights\\legonet_final.pt")
    utils.save_named_module_weights(m,"D:\\PyCharmProjects\\LEGONet\\lean_weights\\")

if __name__ == '__main__':
    main()

