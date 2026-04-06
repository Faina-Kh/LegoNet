import torch
import torch.nn as nn
import torchvision.ops

from legacy_scripts import losses
import anchors

import math, copy

#import legos_2
from legos import legos_3

import util
import config
from efficientdet.models.retinahead import RetinaHead

from torchvision.ops import roi_align

from legonet.myDataloader import UnNormalizer
import numpy as np
import cv2

from itertools import compress




class LEGONet(nn.Module):

    def __init__(self, dataset_train, network_type, backbone_type, num_classes, pretrained = False, num_anchors = 9,
                min_score = 0.05):
        super(LEGONet, self).__init__()

        self.dataset_train = dataset_train
        self.network_type = network_type
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.min_score = min_score

        if self.backbone_type == "ResNetBackboneModule":
            self.backbone_1 = legos_3.ResNetBackboneModule(depth=50, pretrained=pretrained, name='backbone_for_detect')

            in_channels = 256
            self.find_1 = legos_3.FindModule(num_features_in = in_channels, num_classes=num_classes, name='find_for_detect', task='detection')
            self.where = legos_3.WhereModule(network_type, num_features_in = in_channels)

            prior = 0.01
            self.find_1.feature_classification.output.weight.data.fill_(0)
            self.find_1.feature_classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
            self.where.feature_regression.output.weight.data.fill_(0)
            self.where.feature_regression.output.bias.data.fill_(0)

            self.backbone_2 = legos_3.ResNetBackboneModule(depth=50, pretrained=pretrained, name='backbone_for_count')
            self.find_2 = legos_3.FindModule(num_features_in=in_channels, num_classes=num_classes, name='find_for_count', task='counting')
            #self.find_2.feature_classification.output_counting.weight.data.fill_(0)
            #self.find_2.feature_classification.output_counting.bias.data.fill_(-math.log((1.0 - prior) / prior))

        elif self.backbone_type == "EfficientNetBackboneModule":
            self.backbone_1 = legos_3.EfficientNetBackboneModule()

            in_channels = self.backbone_1.W_bifpn
            self.bbox_head = RetinaHead(num_classes=num_classes,
                                        in_channels=in_channels)

        self.FindForCount = legos_3.FindForCount(num_classes)
        self.FatCountingModule = legos_3.FatCountingModule(num_classes)

        self.LeanCountingModule = legos_3.LeanCountingModule(num_classes, inter_losses = config.Counting.inter_losses)

        self.CountWithRegModule = legos_3.CountWithRegModule(num_classes)

        self.roi_align = roi_align

        self.anchors = anchors.Anchors()
        self.regressBoxes = legos_3.BBoxTransform()
        self.clipBoxes = legos_3.ClipBoxes()

        self.focalLoss = losses.FocalLoss()


        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


    def forward(self, inputs):

        if self.network_type != "both":

                if self.training:
                    img_batch, annotations = inputs
                else:
                    img_batch = inputs[0]
        else:

            ########################################################################
            # w = self.find_2.feature_classification.output_counting.weight.data
            # b = self.find_2.feature_classification.output_counting.bias.data
            # print('b', torch.min(b), torch.max(b))
            # print('w', torch.min(w), torch.max(w))
            #########################################################################

            img_batch, annotations, group_idx, do_counting = inputs


        pyramid_feats = self.backbone_1(img_batch)

        anchors = self.anchors(img_batch)


        if self.network_type == "detection":

            if self.training:
                detect_train_inputs = pyramid_feats, anchors, annotations

                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS, classification_loss = self.find_1(detect_train_inputs)
                    regression_loss = self.where(detect_train_inputs)

                ##################################################################################################
                if self.backbone_1.__class__.__name__ == "EfficientNetBackboneModule":
                    outs = self.bbox_head(pyramid_feats)
                    classification = torch.cat([out for out in outs[0]], dim=1)
                    regression = torch.cat([out for out in outs[1]], dim=1)

                ##################################################################################################


                return classification_loss, regression_loss

            else:
                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS_list = self.find_1(pyramid_feats)

                    classifications = []
                    for SFMS in classification_SFMS_list:
                        batch_size, width, height, channels = SFMS['find_maps'].shape
                        out2 = SFMS['find_maps'].view(batch_size, width, height, self.num_anchors, self.num_classes)
                        classifications.append(out2.contiguous().view(batch_size, -1, self.num_classes))

                    classification_vector = torch.cat(classifications, dim=1)
                    regression_vector = self.where(pyramid_feats)

                transformed_anchors = self.regressBoxes(anchors, regression_vector)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
                detection_outputs = self.get_detection_output(transformed_anchors, classification_vector)

                return detection_outputs


        elif self.network_type == "counting_fat":

            p3 = pyramid_feats[0]
            classification = self.FindForCount(p3)


            if self.training:
                count_train_inputs = classification, annotations
                return self.FatCountingModule(count_train_inputs)

            else:
                return self.FatCountingModule(classification)


        elif self.network_type == "counting_lean":
            if self.training:
                find_train_inputs = pyramid_feats, anchors, None

                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS = self.find_1(find_train_inputs)
                    count_input = classification_SFMS, annotations
                    return self.LeanCountingModule(count_input)

            else:
                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS = self.find_1(pyramid_feats)
                    return self.LeanCountingModule(classification_SFMS)


        elif self.network_type == "counting_reg":

            if self.training:
                pyramid_feats[:config.Counting.num_of_pyr_levels], annotations = inputs
            else:
                pyramid_feats[:config.Counting.num_of_pyr_levels] = inputs

            return self.CountWithRegModule(inputs)


        elif self.network_type == "both":

            detection_anns, counting_anns = annotations

            if self.training:

                detect_train_inputs = pyramid_feats, anchors, detection_anns[:,:,:5]

                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS_list, classification_loss = self.find_1(detect_train_inputs)
                    regression_vector, regression_loss = self.where(detect_train_inputs)
            else:
                if self.backbone_type == "ResNetBackboneModule":
                    classification_SFMS_list = self.find_1(pyramid_feats)
                    regression_vector = self.where(pyramid_feats)


            classifications = []
            for SFMS in classification_SFMS_list:
                batch_size, width, height, channels = SFMS['find_maps'].shape
                out2 = SFMS['find_maps'].view(batch_size, width, height, self.num_anchors, self.num_classes)
                classifications.append(out2.contiguous().view(batch_size, -1, self.num_classes))

            classification_vector = torch.cat(classifications, dim=1)

            regression_vector_new = regression_vector.clone()#.detach()
            classification_vector_new = classification_vector.clone()#.detach()

            # if self.training:
            #     total_loss = classification_loss + regression_loss

            if do_counting:

                bbox_crops_list = []
                relevant_points = []
                crops_orig_boxes = []
                Pi_crops_list = []

                counting_outputs = None

                for img_idx in range(img_batch.shape[0]):

                    transformed_anchors = self.regressBoxes(anchors, regression_vector_new[img_idx].unsqueeze(dim=0))
                    transformed_anchors = self.clipBoxes(transformed_anchors, img_batch[img_idx].unsqueeze(dim=0))

                    detection_outputs = self.get_detection_output(transformed_anchors, classification_vector_new[img_idx].unsqueeze(dim=0))

                    # detection eval output:
                    if (detection_outputs[0].cuda()).equal(torch.zeros(0).cuda()) and not config.Detection.USE_PERFECT_DETECTION_MODE:
                        continue

                    else:
                        img_id = self.dataset_train.image_ids[group_idx[img_idx]]
                        img_info = self.dataset_train.img_info[img_id]

                        # please hold
                        #time.sleep(0.1)

                        # continue with counting
                        bbox_pred = detection_outputs[2]#.detach() #b,n,x1,y1,x2,y2 grad=true
                        box_scores = detection_outputs[0]
                        bbox_pred = torch.cat((bbox_pred, box_scores.unsqueeze(-1)), dim=-1)

                        ########################################################################
                        if self.training and config.Detection.USE_PERFECT_DETECTION_MODE:
                            # ToDo - check the issue of using group_idx[img_idx]] vs just img_idx
                            bbox_pred = detection_anns[img_idx, :, :4].cuda()
                            scores = torch.ones(bbox_pred.shape[0], dtype=torch.float32).cuda()
                            bbox_pred = torch.cat((bbox_pred, scores.unsqueeze(-1)), dim=-1)

                        ########################################################################

                        if len(bbox_pred)==0:
                            continue


                        if (self.training and config.Detection.DO_BBOX_AUGMENTATION_FOR_COUNTING): # training and counting augmentation is needed
                            # create new tensor
                            bbox_pred_adjusted = torch.zeros([bbox_pred.shape[0], 5])

                            for box_idx in range(bbox_pred.shape[0]):
                                x1, y1, x2, y2 = bbox_pred[box_idx][0], bbox_pred[box_idx][1], \
                                                 bbox_pred[box_idx][2], bbox_pred[box_idx][3]

                                bbox_pred_adjusted[box_idx][0], \
                                bbox_pred_adjusted[box_idx][1], \
                                bbox_pred_adjusted[box_idx][2], \
                                bbox_pred_adjusted[box_idx][3] = util.augment_bbox_fancy(x1, y1, x2, y2)
                                bbox_pred_adjusted[box_idx][4] = bbox_pred[box_idx][4]

                            bbox_pred_adjusted = bbox_pred_adjusted.cuda()

                        else:
                            # having the option to rescale the bbox during inference according to one rescaling factor - given by BBOX_ADJUSTMENT_RATIO
                            bbox_pred_adjusted = bbox_pred.clone()
                            if config.Detection.BBOX_ADJUSTMENT_RATIO != 1.0: # need to enlarge/shrink the bbox
                                for box_idx in range(bbox_pred_adjusted.shape[0]):
                                    x1, y1, x2, y2 = bbox_pred_adjusted[box_idx][0], bbox_pred_adjusted[box_idx][1], bbox_pred_adjusted[box_idx][2], bbox_pred_adjusted[box_idx][3]

                                    bbox_pred_adjusted[box_idx][0],\
                                    bbox_pred_adjusted[box_idx][1],\
                                    bbox_pred_adjusted[box_idx][2],\
                                    bbox_pred_adjusted[box_idx][3] = util.scale_bbox(x1,y1,x2,y2,config.Detection.BBOX_ADJUSTMENT_RATIO)

                                    # print(bbox_pred_adjusted[box_idx])

                        #####################################################################################################
                        # Get gt points' annotations for training
                        if self.training:
                            point_anns = self.dataset_train.image_data_points_location[img_info['name']]

                            point_anns_copy = []
                            for d in point_anns:
                                point_anns_copy.append(copy.deepcopy(d))

                            points = self.find_points_in_bbox(img_batch[img_idx], point_anns_copy, bbox_pred_adjusted, img_info['scale'])

                            if config.Counting.do_nmcs:
                                non_suppressed_indices = self.nmcs(bbox_pred_adjusted, points)
                                points = list(compress(points, non_suppressed_indices))
                                bbox_pred_adjusted = bbox_pred_adjusted[non_suppressed_indices, :]

                        # get the predicted crops
                        #    in inference - keep all predicted boxes
                        #    in training - keep only those that include points
                        for box_idx in range(bbox_pred_adjusted.shape[0]):
                            if not config.detect_and_count.crop_from_Pi:
                                bbox_crops = self.get_crops(img_batch[img_idx],
                                                            bbox_pred=bbox_pred_adjusted[box_idx].unsqueeze(dim=0),
                                                            anns=detection_anns[img_idx])
                            else:
                                #input to get_crops is pyramid_feats[0] -> need to rescale the bbox coordinates to its size

                                #should be:
                                image_ratio = self.images_ratios(image_shape=(img_batch[img_idx].shape[1:]),
                                    output_shape=(pyramid_feats[0].shape[2],pyramid_feats[0].shape[3]))

                                #wrong:
                                #image_ratio = self.images_ratios(image_shape=(img_info['width'],img_info['height']),
                                #    output_shape=(pyramid_feats[0].shape[2],pyramid_feats[0].shape[3]))

                                new_coords_0 = bbox_pred_adjusted[box_idx][0].clone() * image_ratio[0]
                                new_coords_1 = bbox_pred_adjusted[box_idx][1].clone() * image_ratio[1]
                                new_coords_2 = bbox_pred_adjusted[box_idx][2].clone() * image_ratio[0]
                                new_coords_3 = bbox_pred_adjusted[box_idx][3].clone() * image_ratio[1]

                                new_coords = torch.tensor([new_coords_0, new_coords_1, new_coords_2, new_coords_3], requires_grad=True).cuda()

                                new_detection_anns = torch.cat([detection_anns[img_idx][:,0].clone().unsqueeze(dim=-1)*image_ratio[0],
                                                                   detection_anns[img_idx][:,1].clone().unsqueeze(dim=-1)*image_ratio[1],
                                                                   detection_anns[img_idx][:,2].clone().unsqueeze(dim=-1)*image_ratio[0],
                                                                   detection_anns[img_idx][:,3].clone().unsqueeze(dim=-1)*image_ratio[1]], dim=-1)#.cuda()
                                # the output of get_crops
                                Pi_output_shape = self.image_output_shape(image_shape=(640,640), pyramid_level=3) # returns (80*80)

                                bbox_crops = self.get_crops(pyramid_feats[0].squeeze(dim=0),
                                                          bbox_pred=new_coords.unsqueeze(dim=0),
                                                          anns=new_detection_anns, output_size= Pi_output_shape, #[256,256], #torch.tensor([new_coords[2]-new_coords[0],new_coords[3]-new_coords[1]],dtype=int)
                                                          view_gt=False)

                                # The new images are the resized crops (e.g. 80*80 for P3 for counting)
                                # adjust predicted box coordinates to crops size
                                # new_image_ratio = self.images_ratios(image_shape= (img_info['width'],img_info['height']),
                                #                                      output_shape=Pi_output_shape)
                                #
                                # bbox_pred_adjusted[box_idx][0] = bbox_pred_adjusted[box_idx][0] * new_image_ratio[0]
                                # bbox_pred_adjusted[box_idx][1] = bbox_pred_adjusted[box_idx][1] * new_image_ratio[1]
                                # bbox_pred_adjusted[box_idx][2] = bbox_pred_adjusted[box_idx][2] * new_image_ratio[0]
                                # bbox_pred_adjusted[box_idx][3] = bbox_pred_adjusted[box_idx][3] * new_image_ratio[1]


                            x1 = bbox_pred_adjusted[box_idx, 0]
                            y1 = bbox_pred_adjusted[box_idx, 1]
                            x2 = bbox_pred_adjusted[box_idx, 2]
                            y2 = bbox_pred_adjusted[box_idx, 3]
                            current_score = bbox_pred_adjusted[box_idx, 4] #float(bbox_pred_adjusted[box_idx, 4]) #.cpu())

                            if not self.training:

                                bbox_crops_list.append(
                                    bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).cuda())

                                crops_orig_boxes.append(
                                    [float(x1.cpu()), float(y1.cpu()), float(x2.cpu()), float(y2.cpu()), current_score])
                                # if config.detect_and_count.crop_from_Pi:
                                #     Pi_crops_list.append(Pi_crops)

                            else: # training
                                if not config.detect_and_count.crop_from_Pi:
                                    current_points = points[box_idx]

                                else:
                                    current_points = points[box_idx]
                                    # current_points={}
                                    # current_points['x'] = list(np.array(points[box_idx]['x'])*new_image_ratio[0])
                                    # current_points['y'] = list(np.array(points[box_idx]['y'])*new_image_ratio[1])

                                points_to_view = []

                                # for boxes that include points
                                if len(current_points['x'])>0:
                                    # current_points['x'] = current_points['x'] - (x1.cpu().numpy()) * np.ones(
                                    #     len(current_points['x']))
                                    # current_points['y'] = current_points['y'] - (y1.cpu().numpy()) * np.ones(
                                    #     len(current_points['y']))
                                    #scale_x = bbox_crops.shape[2] / (x2 - x1).cpu().numpy()
                                    #scale_y = bbox_crops.shape[3] / (y2 - y1).cpu().numpy()

                                    current_points['x'] = torch.FloatTensor(current_points['x']) - torch.tensor(x1* torch.ones(len(current_points['x'])), requires_grad=False)
                                    current_points['y'] = torch.FloatTensor(current_points['y']) - torch.tensor(y1 * torch.ones(len(current_points['y'])), requires_grad=False)

                                    scale_x = bbox_crops.shape[2]/(x2-x1)
                                    scale_y = bbox_crops.shape[3] / (y2 - y1)

                                    current_points['x'] = torch.tensor(current_points['x']*scale_x, requires_grad=False)
                                    current_points['y'] = torch.tensor(current_points['y']*scale_y, requires_grad=False)

                                    for i in range(len(current_points['x'])):
                                       points_to_view.append({'x':current_points['x'][i], 'y':current_points['y'][i]})
                                    #self.view_points_on_img(bbox_crops, points_to_view)

                                    relevant_points.append(points_to_view)

                                    bbox_crops_list.append(
                                        bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).cuda())
                                    crops_orig_boxes.append(
                                        [float(x1.cpu()), float(y1.cpu()), float(x2.cpu()), float(y2.cpu()),
                                         current_score])

                including_counting = False
                if len(bbox_crops_list)>0:
                    if self.training:
                        sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points)
                    else:
                        sample_list = self.getitem(bbox_crops=bbox_crops_list)

                    sample = dataloader.kcsv_collater_2(sample_list)

                    if self.training:
                        corrected_counting_anns = sample['points_annot']
                        corrected_counting_anns = [a.cuda() for a in corrected_counting_anns]

                    num_of_crops = sample['img'].shape[0]
                    #print('crops to backbone2:', num_of_crops)
                    if num_of_crops > 1000:
                        a=1
                        # l1 = []
                        # maps_loss = []
                        # counting_outputs=[[], [], [], [], [], [], []]
                        #
                        # for i in range(num_of_crops):
                        #     new_sample_img = sample['img'][i].unsqueeze(dim=0)
                        #     if self.backbone_type == "ResNetBackboneModule":
                        #         bbox_pyramid_feats = self.backbone_2(new_sample_img.cuda())
                        #
                        #     bbox_pyramid_p3 = [bbox_pyramid_feats[0]]  # [p3]
                        #
                        #     if self.training:
                        #         count_train_inputs = bbox_pyramid_p3, \
                        #                              [corrected_counting_anns[1][i].unsqueeze(dim=0),corrected_counting_anns[2][i].unsqueeze(dim=0),
                        #                               corrected_counting_anns[3][i].unsqueeze(dim=0),corrected_counting_anns[4][i].unsqueeze(dim=0),
                        #                               corrected_counting_anns[5][i].unsqueeze(dim=0)]
                        #
                        #         SFMS_lists, cls_output, maps_loss_i = self.find_2(count_train_inputs)
                        #
                        #         count_input = SFMS_lists, cls_output, [corrected_counting_anns[0][i, :].unsqueeze(dim=0),
                        #                                                corrected_counting_anns[1][i].unsqueeze(dim=0),
                        #                                                corrected_counting_anns[2][i].unsqueeze(dim=0),
                        #                                                corrected_counting_anns[3][i].unsqueeze(dim=0),
                        #                                                corrected_counting_anns[4][i].unsqueeze(dim=0),
                        #                                                corrected_counting_anns[5][i].unsqueeze(dim=0)]
                        #
                        #         l1.append(self.LeanCountingModule(count_input))
                        #         maps_loss.append(maps_loss_i)
                        #
                        #         including_counting = True
                        #
                        #     else:
                        #         SFMS_lists, cls_output = self.find_2(bbox_pyramid_p3)
                        #         count_input = SFMS_lists, cls_output
                        #         count_out = self.LeanCountingModule(count_input)
                        #         for i in range(7):
                        #             counting_outputs[i].append(count_out[i])
                        #
                        # if self.training:
                        #     l1=torch.mean(torch.stack(l1))
                        #     maps_loss=torch.mean(torch.stack(maps_loss))
                        #
                        # else:
                        #     for i in range(7):
                        #         counting_outputs[i] =  torch.cat(counting_outputs[i], dim=0)


                    else:

                        if not config.detect_and_count.crop_from_Pi:
                            # img input should be tensor [b,c,h,w]
                            if self.backbone_type == "ResNetBackboneModule":

                               if config.detect_and_count.two_backbones:
                                   bbox_pyramid_feats = self.backbone_2(sample['img'].cuda())

                               elif config.detect_and_count.single_backbone:
                                    bbox_pyramid_feats = self.backbone_1(sample['img'].cuda())


                            bbox_pyramid_p3 = [bbox_pyramid_feats[0]]  # [p3]

                        else: #currently use crop only from P3
                            #bbox_pyramid_p3 = [torch.cat(Pi_crops_list, dim=0)]
                            bbox_pyramid_p3 = [torch.cat(bbox_crops_list, dim=0)]

                        # if self.backbone_type == "ResNetBackboneModule":
                        #     bbox_pyramid_feats = self.backbone_2(sample['img'].cuda())

                        if self.training:

                            if config.Counting.counting_type == 'withKeyPoints':
                                count_train_inputs = bbox_pyramid_p3, corrected_counting_anns[1:6]  # anchors, None
                                SFMS_lists, cls_output, maps_loss = self.find_2(count_train_inputs)
                                count_input = SFMS_lists, cls_output, corrected_counting_anns

                                l1 = self.LeanCountingModule(count_input)


                                if config.detect_and_count.balance_losses:
                                    maps_loss=maps_loss/1000
                                    l1=l1/100

                                counting_loss = l1 + maps_loss


                            elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                                if not config.detect_and_count.crop_from_Pi:
                                    counting_loss = self.CountWithRegModule([bbox_pyramid_feats[:config.Counting.num_of_pyr_levels], corrected_counting_anns[0]])
                                else:
                                    counting_loss = self.CountWithRegModule([bbox_pyramid_p3,corrected_counting_anns[0]])

                            #total_loss = total_loss + counting_loss

                            including_counting = True

                        else:
                            if config.Counting.counting_type == 'withKeyPoints':
                                SFMS_lists, cls_output = self.find_2(bbox_pyramid_p3)
                                count_input = SFMS_lists, cls_output
                                counting_outputs = self.LeanCountingModule(count_input)

                            elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                                # counting_outputs = self.CountWithRegModule(bbox_pyramid_feats[:config.Counting.num_of_pyr_levels])

                                if not config.detect_and_count.crop_from_Pi:
                                    counting_outputs = self.CountWithRegModule(bbox_pyramid_feats[:config.Counting.num_of_pyr_levels])
                                else:
                                    counting_outputs = self.CountWithRegModule(bbox_pyramid_p3)


            else:
                if not self.training:
                    #ToDo - fix this to work with img_batch>0 - currently returns detection_outputs for the last img becouse assumes one image
                    for img_idx in range(img_batch.shape[0]):
                        transformed_anchors = self.regressBoxes(anchors,
                                                                regression_vector_new[img_idx].unsqueeze(dim=0))
                        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch[img_idx].unsqueeze(dim=0))

                        detection_outputs = self.get_detection_output(transformed_anchors,
                                                                      classification_vector_new[img_idx].unsqueeze(
                                                                          dim=0))

            if self.training:
                if do_counting:
                    if not including_counting:
                        if config.Counting.counting_type == 'withKeyPoints':
                            return classification_loss, regression_loss, None, None
                        elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                            return classification_loss, regression_loss, None

                    if config.Counting.counting_type == 'withKeyPoints':
                        return classification_loss, regression_loss, l1, maps_loss  #counting_loss #total_loss #classification_loss, regression_loss, counting_loss

                    elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                        return classification_loss, regression_loss, counting_loss

                else:
                    if config.Counting.counting_type == 'withKeyPoints':
                        return classification_loss, regression_loss, None, None
                    elif config.Counting.counting_type == 'reg_fpn_p3_p7_min_sig':
                        return classification_loss, regression_loss, None

            else:
                if len(bbox_crops_list)>0:
                    return detection_outputs, counting_outputs, sample, relevant_points, crops_orig_boxes #bbox_pred_adjusted,....
                else:
                    return detection_outputs, counting_outputs, None, None, None  #[],...





    def get_detection_output(self, transformed_anchors, classification_vector):

        scores = torch.max(classification_vector, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > self.min_score)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification_vector[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        if config.detect_and_count.cancel_nms_in_train:
            current_scores, current_classes = classification[0, :, :].max(dim=1)
            return [current_scores, current_classes, transformed_anchors[0, :, :]]

        else:
            anchors_nms_idx = torchvision.ops.nms(transformed_anchors[0, :, :], scores[0, :, 0], config.Detection.NMS_THRESHOLD)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def get_crops(self, img, bbox_pred=None, anns=None, view_gt = False, output_size=(config.Counting.crops_size[0], config.Counting.crops_size[1])):
        if view_gt:
            unnormalize = UnNormalizer()

            im = img.cpu().clone()#.detach()
            im = np.array(255 * unnormalize(im))
            im[im < 0] = 0
            im[im > 255] = 255
            im = np.transpose(im, (1,2,0))
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imshow('img', im)
            cv2.waitKey(0)

            #draw gt boxes for each img - based on anns
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

            im=torch.tensor(im).float().permute(2,0,1).unsqueeze(dim=0).cuda()


        bbox_crops = self.roi_align(input=img.unsqueeze(dim=0), boxes=[bbox_pred], output_size=output_size)
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
        return output_shape /np.array(image_shape[:2]) #torch.tensor([output_shape[0],output_shape[1]], requires_grad=False)/torch.tensor([image_shape[0],image_shape[1]],requires_grad=False)

    # spezold
    from typing import Union
    def percentile(self, t: torch.tensor, q: float) -> Union[int, float]:
        """
        Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.

        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result


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

        x2 = (((torch.tensor(x, dtype=float, requires_grad=False).cuda() - torch.round(center_point[0])) * s) / radius[0]) ** 2  #(((x - round(center_point[0])) * s) / radius[0]) ** 2
        y2 = (((torch.tensor(y, dtype=float, requires_grad=False).cuda() - torch.round(center_point[1])) * s) / radius[1]) ** 2

        p = torch.exp(-0.5 * (x2 + y2)) #np.exp(-0.5 * (x2 + y2))

        p[torch.where(p < self.percentile(p, q))] = 0 #p[np.where(p < np.percentile(p, q))] = 0

        p = p /torch.max(p)   #p / np.max(p)
        #if not np.isfinite(p).all() or not np.isfinite(p).all():
         #   print('divide by zero')
        if not torch.isfinite(p).all() or not torch.isfinite(p).all():
            print('divide by zero')


        return p


    def compute_keypoints_targets_multi_maps(self, image_shape, annotations_points_centers_a, radius=(5, 5), pyramid_level=3):
        # resize transformed-image and annotations
        annotations_points_centers = annotations_points_centers_a.copy()  #copy.deepcopy(annotations_points_centers_a)
        # here we should resize image too and then check it with the annotations

        if not config.detect_and_count.crop_from_Pi:
            output_shape = self.image_output_shape(image_shape, pyramid_level=pyramid_level)

        else:
            output_shape = image_shape[:2]

        image_ratio = self.images_ratios(image_shape, output_shape)

        if len(annotations_points_centers) == 0:
            return [torch.zeros(output_shape,requires_grad=False).cuda()] #[np.zeros(output_shape)]

        annotations = torch.zeros(output_shape,requires_grad=False).cuda() #np.zeros(output_shape)
        for i in range(len(annotations_points_centers)):
            annotations_points_centers[i]['y'] *= image_ratio[0]
            annotations_points_centers[i]['x'] *= image_ratio[1]

            annotations_points_centers[i]['y'].cuda()
            annotations_points_centers[i]['x'].cuda()


            current_points = [annotations_points_centers[i]['x'], annotations_points_centers[i]['y'] ]
            gaussian_map = self.create_gausian_mask(current_points[:2], output_shape[1], output_shape[0],
                                                    radius=radius)
            # each center point in the GT will be 1 in the annotation map
            gaussian_map=gaussian_map.type(torch.float)
            annotations = torch.max(annotations, gaussian_map) #np.maximum(annotations, gaussian_map)

            if torch.isnan(annotations).any(): #np.isnan(annotations).any():
                raise ("nan was found")

        return annotations


    def getitem(self, bbox_crops, points = None):
        filtered_samples = []
        for b in range(len(bbox_crops)):
            current_img = bbox_crops[b][0].permute(1,2,0).cpu()
            sample = {'img': current_img}

            #sample['lean_version'] = self.lean_version

            if points:
                current_ann = points[b]
                sample['annot'] = current_ann


                if len(current_ann) > 0:
                    annotations_group_num_of_points =  len(current_ann)
                    annotations_group_points_center = current_ann

                    annotation_map_1 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.Counting.map_1_R) #(5,7))
                    annotation_map_2 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.Counting.map_2_R) #(3, 5))
                    annotation_map_3 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius= config.Counting.map_3_R) #(3, 7))
                    annotation_map_4 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.Counting.map_4_R) #(5, 5))
                    annotation_map_5 = self.compute_keypoints_targets_multi_maps(sample['img'].shape,
                                                                                 annotations_group_points_center,
                                                                                 radius=config.Counting.map_5_R) #(3,3)

                    # plt.imsave('vis' + '/' + 'ann1' + '_Relu.png', annotation_map_1)
                    # plt.imsave('vis' + '/' + 'ann2' + '_Relu.png', annotation_map_2)
                    # plt.imsave('vis' + '/' + 'ann3' + '_Relu.png', annotation_map_3)
                    # plt.imsave('vis' + '/' + 'ann4' + '_Relu.png', annotation_map_4)
                    # plt.imsave('vis' + '/' + 'ann5' + '_Relu.png', annotation_map_5)


                    sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                                       annotation_map_3, annotation_map_4, annotation_map_5]


            filtered_samples.append(sample)

        return filtered_samples


    def find_points_in_bbox(self, img, point_anns, bbox_pred, scale):

        unnormalize = UnNormalizer()

        points=[]

        # for drawing
        im = img.cpu().clone()#.detach()
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
            box_x1, box_y1, box_x2, box_y2 = bbox_pred[b,:4]
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
    util.save_named_module_weights(m,"D:\\PyCharmProjects\\LEGONet\\lean_weights\\")

if __name__ == '__main__':
    main()

