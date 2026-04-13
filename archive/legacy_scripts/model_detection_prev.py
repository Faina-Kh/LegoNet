import torch
import torch.nn as nn
import torchvision.ops

from legacy_scripts import losses
import anchors

import math, copy

from legos import legos_3

import util
import config
from efficientdet.models.retinahead import RetinaHead

from torchvision.ops import roi_align

from legonet.myDataloader import UnNormalizer
import numpy as np
import cv2

from itertools import compress
from legacy_eval.both_eval_new import choose_boxes_by_IoUandPrc





class LEGONet(nn.Module):

    def __init__(self, dataset_train, network_type, backbone_type, num_classes, pretrained = False, num_anchors = 9,
                min_score = 0.05, output_size = 1, version="", separate_training = False , device = None):
        super(LEGONet, self).__init__()

        self.dataset_train = dataset_train
        self.network_type = network_type
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.min_score = min_score

        self.separate_training = separate_training

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

            if config.General.twoFind_2:
                self.find_2_b = legos_3.FindModule(num_features_in=in_channels, num_classes=num_classes, name='find_for_count', task='counting')
            if config.General.twoBackbone_2:
                self.backbone_2_b = legos_3.ResNetBackboneModule(depth=50, pretrained=pretrained, name='backbone_for_count')

            if config.Detect_and_Estimate.use_new_Find:
                self.find_2_length = legos_3.FindModule(num_features_in=in_channels, num_classes=num_classes, name='find_for_count', task='counting')
                self.find_2_diameter = legos_3.FindModule(num_features_in=in_channels, num_classes=num_classes, name='find_for_count', task='counting')
                self.find_2_color = legos_3.FindModule(num_features_in=in_channels, num_classes=num_classes, name='find_for_count', task='counting')


            #self.find_2.feature_classification.output_counting.weight.data.fill_(0)
            #self.find_2.feature_classification.output_counting.bias.data.fill_(-math.log((1.0 - prior) / prior))

        elif self.backbone_type == "EfficientNetBackboneModule":
            self.backbone_1 = legos_3.EfficientNetBackboneModule()

            in_channels = self.backbone_1.W_bifpn
            self.bbox_head = RetinaHead(num_classes=num_classes,
                                        in_channels=in_channels)

        self.FindForCount = legos_3.FindForCount(num_classes)
        self.FatCountingModule = legos_3.FatCountingModule(num_classes)

        if self.network_type == "both_for_roots_2":
             config.General.with_new_layers_for_both = True

        self.LeanCountingModule = legos_3.LeanCountingModule(num_classes, inter_losses = config.AttributeEstimation.inter_losses)

        if self.network_type == "both_for_roots_2" or config.General.binary_model: #config.General.with_new_layers:
            config.General.binary_model = True
            self.LeanCountingModule_color = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, loss_for ="color")

            config.General.binary_model = False
            self.LeanCountingModule_length = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, loss_for ="length")
            self.LeanCountingModule_diameter = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, loss_for ="diameter")

        self.LeanCountingModule_multiple = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, output_size = output_size)
        self.LeanCountingModule_multiple_2 = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, output_size=output_size)
        self.LeanCountingModule_multiple_3 = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, output_size=output_size)
        self.LeanCountingModule_multiple_4 = legos_3.LeanCountingModule(num_classes, inter_losses=config.AttributeEstimation.inter_losses, output_size=output_size)

        self.CountWithRegModule = legos_3.CountWithRegModule(num_classes)

        if self.network_type == "both_for_roots":
            self.LeanModuleForRoots = legos_3.LeanModuleForRoots(num_classes, inter_losses = config.AttributeEstimation.inter_losses)

        if self.network_type == "both_for_roots_2" and config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
            self.CountWithRegModule_color = legos_3.CountWithRegModule(num_classes)
            self.CountWithRegModule_length = legos_3.CountWithRegModule(num_classes)
            self.CountWithRegModule_diameter = legos_3.CountWithRegModule(num_classes)

        self.roi_align = roi_align

        if config.Detection.change_anchors:
            self.anchors = anchors.Anchors(ratios=config.Detection.ratios)
        else:
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

        if self.network_type != "both" and self.network_type != "both_for_roots" and self.network_type != "both_for_roots_2":

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

        anchors = self.anchors(img_batch)


        if self.network_type == "detection":

            pyramid_feats = self.backbone_1(img_batch)

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

            pyramid_feats = self.backbone_1(img_batch)

            p3 = pyramid_feats[0]
            classification = self.FindForCount(p3)


            if self.training:
                count_train_inputs = classification, annotations
                return self.FatCountingModule(count_train_inputs)

            else:
                return self.FatCountingModule(classification)


        elif self.network_type == "counting_lean":
            # if self.training:
            #     find_train_inputs = pyramid_feats, anchors, None
            #
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(find_train_inputs)
            #         count_input = classification_SFMS, annotations
            #         return self.LeanCountingModule(count_input)
            #
            # else:
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(pyramid_feats)
            #         return self.LeanCountingModule(classification_SFMS)

            pyramid_feats = self.backbone_1(img_batch)

            p3 = pyramid_feats[0]

            if self.training:
                for i in range(6):
                    annotations[i] = annotations[i].device #cuda()
                count_train_inputs = [p3], annotations[1:6] # anchors, None
                SFMS_lists, cls_output, maps_loss = self.find_2(count_train_inputs)
                count_input = SFMS_lists, cls_output, annotations

                l1 = self.LeanCountingModule(count_input)
                return l1[0], maps_loss
                #counting_loss = l1 + maps_loss
                #return counting_loss

            else:

                SFMS_lists, cls_output = self.find_2([p3])
                count_input = SFMS_lists, cls_output
                counting_outputs = self.LeanCountingModule(count_input)
                return counting_outputs


        elif self.network_type == "counting_lean_multiple_out":
            # if self.training:
            #     find_train_inputs = pyramid_feats, anchors, None
            #
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(find_train_inputs)
            #         count_input = classification_SFMS, annotations
            #         return self.LeanCountingModule(count_input)
            #
            # else:
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(pyramid_feats)
            #         return self.LeanCountingModule(classification_SFMS)

            pyramid_feats = self.backbone_1(img_batch)

            p3 = pyramid_feats[0]

            if self.training:
                for i in range(6):
                    annotations[i] = annotations[i].device #cuda()

                train_inputs = [p3], annotations[1:6] # anchors, None

                SFMS_lists, cls_output, maps_loss = self.find_2(train_inputs)

                new_input = SFMS_lists, cls_output, annotations

                l1 , l2, l3, l4 = self.LeanCountingModule_multiple(new_input)

                return l1, l2, l3, l4, maps_loss
                #counting_loss = l1 + maps_loss
                #return counting_loss

            else:

                SFMS_lists, cls_output = self.find_2([p3])
                new_input = SFMS_lists, cls_output
                new_outputs = self.LeanCountingModule_multiple(new_input)
                return new_outputs


        elif self.network_type == "counting_lean_multiple_out_V2":
            # if self.training:
            #     find_train_inputs = pyramid_feats, anchors, None
            #
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(find_train_inputs)
            #         count_input = classification_SFMS, annotations
            #         return self.LeanCountingModule(count_input)
            #
            # else:
            #     if self.backbone_type == "ResNetBackboneModule":
            #         classification_SFMS = self.find_1(pyramid_feats)
            #         return self.LeanCountingModule(classification_SFMS)

            pyramid_feats = self.backbone_1(img_batch)

            p3 = pyramid_feats[0]

            if self.training:
                for i in range(6):
                    annotations[i] = annotations[i].cuda()

                train_inputs = [p3], annotations[1:6] # anchors, None

                SFMS_lists, cls_output, maps_loss = self.find_2(train_inputs)

                new_input_1 = SFMS_lists, cls_output, [annotations[0][:,:,0], annotations[1], annotations[2], annotations[3], annotations[4], annotations[5]]
                new_input_2 = SFMS_lists, cls_output, [annotations[0][:,:,1], annotations[1], annotations[2], annotations[3], annotations[4], annotations[5]]
                new_input_3 = SFMS_lists, cls_output, [annotations[0][:,:,2], annotations[1], annotations[2], annotations[3], annotations[4], annotations[5]]
                new_input_4 = SFMS_lists, cls_output, [annotations[0][:,:,3], annotations[1], annotations[2], annotations[3], annotations[4], annotations[5]]

                l1  = self.LeanCountingModule_multiple(new_input_1)

                l2 = self.LeanCountingModule_multiple_2(new_input_2)

                l3 = self.LeanCountingModule_multiple_3(new_input_3)

                l4 = self.LeanCountingModule_multiple_4(new_input_4)

                return l1[0], l2[0], 10*l3[0], 100*l4[0], maps_loss
                #counting_loss = l1 + maps_loss
                #return counting_loss

            else:

                SFMS_lists, cls_output = self.find_2([p3])
                new_input = SFMS_lists, cls_output
                output1, map0, map1, map2, map3, map4, map5 = self.LeanCountingModule_multiple(new_input)
                output2, map0, map1, map2, map3, map4, map5 = self.LeanCountingModule_multiple_2(new_input)
                output3, map0, map1, map2, map3, map4, map5 = self.LeanCountingModule_multiple_3(new_input)
                output4, map0, map1, map2, map3, map4, map5 = self.LeanCountingModule_multiple_4(new_input)

                return [[output1,output2,output3,output4], map0, map1, map2, map3, map4, map5 ]


        elif self.network_type == "both" or self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":


            if not self.separate_training:
                pyramid_feats = self.backbone_1(img_batch)

            if config.detect_with_points.detect_points:
                p3 = pyramid_feats[0]
                detection_anns, counting_anns, per_obj_maps = annotations
                #anno = per_obj_maps[1][1][0]
                #plt.imsave("D:\\Faina\\roots_project\\anno_2_18_12.png", anno)

            else:
                if annotations is None:
                    detection_anns, counting_anns = None, None
                else:
                    detection_anns, counting_anns = annotations

            # for both_for_roots
            # detection_anns: [batch,box, x1,y1,x2,y2,class,box_id]
            # counting_anns: [batch][0]: [count,class, box_id,length,diameter]
            #                       [1]: [point_x, point_y,class, box_id]

            if self.training:

                detect_train_inputs = pyramid_feats, anchors, detection_anns #detection_anns[:,:,:5]

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

            regression_vector_new = regression_vector.clone().detach()
            classification_vector_new = classification_vector.clone().detach()

            # if self.training:
            #     total_loss = classification_loss + regression_loss

            if do_counting: #and counting_anns is not None:

                bbox_crops_list = []
                relevant_points = []
                crops_orig_boxes = []

                counting_outputs = None

                for img_idx in range(img_batch.shape[0]):

                    transformed_anchors = self.regressBoxes(anchors, regression_vector_new[img_idx].unsqueeze(dim=0))
                    transformed_anchors = self.clipBoxes(transformed_anchors, img_batch[img_idx].unsqueeze(dim=0))

                    detection_outputs = self.get_detection_output(transformed_anchors, classification_vector_new[img_idx].unsqueeze(dim=0))

                    # detection eval output:
                    if (detection_outputs[0].cuda()).equal(torch.zeros(0).cuda()) and not config.Detection.USE_PERFECT_DETECTION_MODE:
                        bbox_pred = []
                        bbox_pred_adjusted=torch.empty(0)
                        continue

                    else:
                        img_id = self.dataset_train.image_ids[group_idx[img_idx]]
                        img_info = self.dataset_train.img_info[img_id]

                        # please hold
                        #time.sleep(0.1)

                        # continue with counting
                        bbox_pred = detection_outputs[2].detach() #b,n,x1,y1,x2,y2 grad=true
                        box_scores = detection_outputs[0]


                        if counting_anns is not None:
                            # find the gt box for each detection
                            pred_to_box_idx = choose_boxes_by_IoUandPrc(bbox_pred, detection_anns, box_scores)

                        if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":

                            if (self.training or (not self.training and annotations is not None)) and counting_anns is not None: # and config.detect_with_points.detect_points: #

                                ##############################################################################################
                                for box in pred_to_box_idx:
                                    # box ids start with 1, not 0
                                    if box[:, 4] != -1:
                                        box[:, 4] = box[:, 4] + 1

                                bbox_pred = torch.cat(pred_to_box_idx, dim=0)

                                if self.training:
                                    # keep only the annotations of detected boxes
                                    box_scores = box_scores[bbox_pred[:, 4] != -1]
                                    bbox_pred = bbox_pred[bbox_pred[:, 4] != -1]

                                true_ids = bbox_pred[:, 4].sort().values

                                if not self.training:
                                    true_ids=true_ids.cpu()

                                #img_batch, annotations, group_idx, do_counting = inputs
                                #detection_anns, counting_anns, per_obj_maps = annotations

                                anns_box_ids = detection_anns[0][:,5]
                                ans_detected_ids = []
                                for i in range(anns_box_ids.shape[0]):
                                    ans_detected_ids.append(anns_box_ids[i] in true_ids)

                                detection_anns = detection_anns[0][ans_detected_ids].unsqueeze(dim=0)

                                counting_anns[0][0] = counting_anns[0][0][ans_detected_ids]

                                anns_box_ids_p = counting_anns[0][1][:,3].float()
                                ans_detected_ids_p = []
                                for i in range(counting_anns[0][1].shape[0]):
                                    ans_detected_ids_p.append(anns_box_ids_p[i] in true_ids.cpu())

                                counting_anns[0][1] = counting_anns[0][1][ans_detected_ids_p]

                                if config.detect_with_points.detect_points:
                                    per_obj_maps[0] = true_ids
                                    per_obj_maps[1] = per_obj_maps[1][ans_detected_ids]
                                    annotations = [detection_anns, counting_anns, per_obj_maps]

                                else:
                                    annotations = [detection_anns, counting_anns]


                        bbox_pred = torch.cat((bbox_pred, box_scores.unsqueeze(-1)), dim=-1) #[x1,y1,x2,y2,gt_box_id,score]

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
                                bbox_pred_adjusted[box_idx][3] = util.augment_bbox_fancy(x1, y1, x2, y2) #util.augment_bbox(x1, y1, x2, y2)   #util.augment_bbox_fancy(x1, y1, x2, y2)
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
                        points=None
                        if self.training or (not self.training and annotations is not None):
                            point_anns = self.dataset_train.image_data_points_location[img_info['name']]
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

                                if not config.detect_with_points.detect_points:
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

                                            if not config.detect_with_points.detect_points:
                                                current_points['x'] = current_points['x'] - (x1.cpu().numpy()) * np.ones(len(current_points['x']))
                                                current_points['y'] = current_points['y'] - (y1.cpu().numpy()) * np.ones(len(current_points['y']))

                                                scale_x = config.AttributeEstimation.crops_size[0] / (x2 - x1).cpu().numpy()
                                                scale_y = config.AttributeEstimation.crops_size[1] / (y2 - y1).cpu().numpy()
                                                current_points['x'] = current_points['x']*scale_x
                                                current_points['y'] = current_points['y']*scale_y

                                                for i in range(len(current_points['x'])):
                                                   points_to_view.append({'x':current_points['x'][i], 'y':current_points['y'][i]})
                                                #self.view_points_on_img(bbox_crops, points_to_view)

                                                relevant_points.append(points_to_view)

                                            if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                                                if not config.detect_with_points.detect_points:
                                                    bbox_crops_list.append([bbox_crops, bbox_pred_adjusted[b, 4]]) # add the gt box id
                                            else:
                                                bbox_crops_list.append(bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).cuda())

                                        else:
                                            #empty_crops_gtidx.append(box_idx)

                                            if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                                                if not self.training:
                                                    bbox_crops_list.append([bbox_crops, torch.tensor(-1, dtype=float)])
                                                    relevant_points.append([])

                                    else:
                                        bbox_crops_list.append(bbox_crops)

                                else:
                                    if not config.detect_with_points.detect_points:
                                        bbox_crops_list.append(bbox_crops)  # torch.tensor(bbox_crops).float().permute(2, 0, 1).unsqueeze(dim=0).cuda())



                            # if len(empty_crops_gtidx)>0:
                            #     bbox_pred_adjusted_2 = []
                            #     for b in range(bbox_pred_adjusted.shape[0]):
                            #         box_idx = int(bbox_pred_adjusted[b, 4].cpu())
                            #         if box_idx not in empty_crops_gtidx:
                            #             bbox_pred_adjusted_2.append(bbox_pred_adjusted[b].unsqueeze(dim=0))
                            #
                            #     bbox_pred_adjusted = torch.cat(bbox_pred_adjusted_2, dim=0)
                            #
                            #     keep_crops = []
                            #     for i in range(annotations[0].shape[1]):
                            #         keep_crops.append(annotations[0][:, i, 5] not in empty_crops_gtidx)
                            #
                            #     annotations[0] = annotations[0][:,keep_crops,:]
                            #     annotations[1][0][0] = annotations[1][0][0][keep_crops,:]
                            #
                            #     keep_points = []
                            #     for i in range(annotations[1][0][1].shape[0]):
                            #         keep_points.append(annotations[1][0][1][i,3] not in empty_crops_gtidx)
                            #
                            #     annotations[1][0][1] = annotations[1][0][1][keep_points]


                including_counting = False

                if len(bbox_crops_list)>0 or (config.detect_with_points.detect_points and len(bbox_pred)>0):

                    num_of_boxes = bbox_pred_adjusted.shape[0]

                    if not config.detect_with_points.detect_points:
                        if self.training:
                            if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                                 sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points, anns = annotations)
                            else:
                                sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points)
                        else:
                            if annotations is not None and points is not None:
                                sample_list = self.getitem(bbox_crops=bbox_crops_list, points=relevant_points,
                                                           anns=annotations)
                            else:
                                sample_list = self.getitem(bbox_crops=bbox_crops_list)

                        sample = dataloader.kcsv_collater_2(sample_list)

                        if self.training or (not self.training and annotations is not None and points is not None):
                            corrected_counting_anns = sample['points_annot']
                            corrected_counting_anns = [a.cuda() for a in corrected_counting_anns]

                            if config.detect_with_points.detect_points:
                                corrected_maps_anns = sample['box_maps_annot']
                                corrected_maps_anns = [a.cuda() for a in corrected_maps_anns]


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
                            #torch.empty_cache()
                            if not config.detect_with_points.detect_points:
                                # img input should be tensor [b,c,h,w]
                                if self.backbone_type == "ResNetBackboneModule":
                                    if config.Detect_and_Estimate.single_backbone:

                                        bbox_pyramid_feats = self.backbone_1(sample['img'].cuda())

                                    else:
                                        bbox_pyramid_feats= []
                                        bbox_pyramid_p3 = []

                                        if config.General.twoBackbone_2:
                                            bbox_pyramid_feats_2 = []
                                            bbox_pyramid_p3_2 = []

                                        for i in range(num_of_crops):
                                            current= self.backbone_2(sample['img'][i].unsqueeze(dim=0).cuda())
                                            bbox_pyramid_feats.append(current)
                                            if i==0:
                                                bbox_pyramid_p3.append(bbox_pyramid_feats[i][0]) ##current[0]
                                            else:
                                                bbox_pyramid_p3[0] = torch.cat((bbox_pyramid_p3[0], bbox_pyramid_feats[i][0]), dim=0)  #[bbox_pyramid_feats[0]]  # [bbox_pyramid_feats]  # [p3]

                                            if config.General.twoBackbone_2:
                                                current_2 = self.backbone_2_b(sample['img'][i].unsqueeze(dim=0).cuda())
                                                bbox_pyramid_feats_2.append(current_2)
                                                if i == 0:
                                                    bbox_pyramid_p3_2.append(bbox_pyramid_feats_2[i][0])
                                                else:
                                                    bbox_pyramid_p3_2[0] = torch.cat((bbox_pyramid_p3_2[0], bbox_pyramid_feats_2[i][0]),dim=0)



                                        # bbox_pyramid_feats = torch.cat(bbox_pyramid_feats, dim=0)

                    else:

                        # image_ratio = self.images_ratios(image_shape=(img_batch[img_idx].shape[1:]),
                        #                                  output_shape=(
                        #                                  pyramid_feats[0].shape[2], pyramid_feats[0].shape[3]))
                        # new_coords_0 = bbox_pred_adjusted[box_idx][0].clone() * image_ratio[0]
                        # new_coords_1 = bbox_pred_adjusted[box_idx][1].clone() * image_ratio[1]
                        # new_coords_2 = bbox_pred_adjusted[box_idx][2].clone() * image_ratio[0]
                        # new_coords_3 = bbox_pred_adjusted[box_idx][3].clone() * image_ratio[1]
                        #
                        # new_coords = torch.tensor([new_coords_0, new_coords_1, new_coords_2, new_coords_3],
                        #                           requires_grad=True).cuda()
                        #

                        # maps = []
                        # for box_num in range(num_of_crops):
                        #     new_p3 = p3.clone()
                        #     current_coord = bbox_pred_adjusted[box_num][:4]
                        #     w = new_p3.shape[2]
                        #     h = new_p3.shape[3]
                        #     for i in range(w):
                        #         for j in range(h):
                        #             x1 = torch.floor(current_coord[0])
                        #             y1 = torch.floor(current_coord[1])
                        #             x2 = torch.ceil(current_coord[2])
                        #             y2 = torch.ceil(current_coord[3])
                        #
                        #             if not (i > x1 and i < x2 and j > y1 and j < y2):
                        #                 new_p3[:, :, i, j] = 0
                        #
                        #     maps.append(new_p3)
                        #
                        # p3 = torch.cat(maps, dim=0)

                        new_im = []
                        #num_of_boxes = bbox_pred_adjusted.shape[0]

                        p3 = []
                        ########################
                        # count_loss = 0
                        # length_loss=0
                        # diameter_loss=0
                        #######################
                        for box_num in range(num_of_boxes):
                            img_batch_copy = img_batch.clone()

                            # visualize orig
                            #unnormalize = UnNormalizer()

                            # save_orig = os.path.join("D:\\Faina\\roots_project", "Dataset for root color model\\17_1_data\\" , \
                            #         "For root color model\\all images\\Results\\both_detect by points\\val_2023-01-23_223454_epoch_299\\test", "orig.jpg")
                            # orig = img_batch_copy[0].cpu().clone().detach()
                            #
                            # orig = np.array(255 * unnormalize(orig))
                            # orig[orig < 0] = 0
                            # orig[orig > 255] = 255
                            # orig = np.transpose(orig, (1, 2, 0))
                            # orig = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_BGR2RGB)
                            # #cv2.imshow('img2', orig)
                            # #cv2.waitKey(0)
                            # cv2.imwrite(save_orig, orig)

                            current_coord = bbox_pred_adjusted[box_num][:4]
                            x1 = int(torch.floor(current_coord[0]).item())
                            y1 = int(torch.floor(current_coord[1]).item())
                            x2 = int(torch.ceil(current_coord[2]).item())
                            y2 = int(torch.ceil(current_coord[3]).item())

                            #wrong!!!!!
                            # img_batch_copy[:, :, :(x1+1), :] = 0
                            # img_batch_copy[:, :, x2:, :] = 0
                            # img_batch_copy[:, :,:, :(y1+1)] = 0
                            # img_batch_copy[:, :, :, y2:] = 0

                            img_batch_copy[:, :, :(y1 + 1), :] = 0
                            img_batch_copy[:, :, y2:, :] = 0
                            img_batch_copy[:, :, :, :(x1 + 1)] = 0
                            img_batch_copy[:, :, :, x2:] = 0

                            #torch.empty_cache()
                            bbox_pyramid_feats_new = self.backbone_2(img_batch_copy)

                            p3.append(bbox_pyramid_feats_new[0])


                            #new_im.append(img_batch_copy)

                            # visualize
                            # im = img_batch_copy[0].cpu().clone().detach()
                            # im = np.array(255 * unnormalize(im))
                            # im[im < 0] = 0
                            # im[im > 255] = 255
                            # im = np.transpose(im, (1, 2, 0))
                            # im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)
                            # #cv2.imshow('img', im)
                            # #cv2.waitKey(0)
                            # save_path = "D:\\Faina\\roots_project\\Dataset for root color model\\17_1_data\\" +\
                            #             "For root color model\\all images\\Results\\both_detect by points\\val_2023-01-23_223454_epoch_299\\test//crop_"+ str(box_num)+".jpg"
                            # cv2.imwrite(save_path, im)


                        #new_im = torch.cat(new_im, dim=0)
                        #bbox_pyramid_feats_new = self.backbone_2(img_batch_copy)   #self.backbone_2(new_im)

                        #p3 = [bbox_pyramid_feats_new[0]]  # [p3]

                            # if config.Counting.counting_type == 'withKeyPoints' and config.detect_with_points.detect_points:
                            #
                            #     root_anns = annotations[1][0][0].cuda().float()
                            #     annotations[1][0][1] = annotations[1][0][1].cuda().float()
                            #     annotations[2][1] = annotations[2][1].cuda().float()
                            #
                            #     count_train_inputs = p3, annotations[2][1]  # corrected_maps_anns
                            #
                            #     SFMS_lists, cls_output, maps_loss = self.find_2(count_train_inputs)
                            #     count_input = SFMS_lists, cls_output, root_anns
                            #
                            #     current_count_loss, current_length_loss, current_diameter_loss = self.LeanModuleForRoots(count_input)
                            #     count_loss += current_count_loss
                            #     length_loss += current_length_loss
                            #     diameter_loss += current_diameter_loss


                        #bbox_pyramid_feats_new =  torch.cat(bbox_pyramid, dim=0)     #self.backbone_2(img_batch_copy)#(new_im)

                        p3 = [torch.cat(p3, dim=0)]  #[bbox_pyramid_feats_new[0]]  # [p3]


                    if self.training:

                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                            if not config.detect_with_points.detect_points:
                                count_train_inputs = bbox_pyramid_p3, corrected_counting_anns[1:6]  # anchors, None
                                root_anns = corrected_counting_anns[6:9]

                            else:
                                root_anns = annotations[1][0][0].cuda().float()
                                annotations[1][0][1] = annotations[1][0][1].cuda().float()
                                annotations[2][1] = annotations[2][1].cuda().float()

                                count_train_inputs = p3, annotations[2][1][:,0,:,:] #corrected_maps_anns


                            l1 = 0 # for "both"

                            maps_loss = 0
                            count_loss = 0
                            length_loss = 0
                            diameter_loss = 0

                            color_loss = 0

                            for i in range(num_of_boxes):

                                current_roots_anns = [root_anns[0][i][0].unsqueeze(dim=0),root_anns[0][i][1].unsqueeze(dim=0),root_anns[0][i][2].unsqueeze(dim=0)]
                                #[root_anns[0][i].unsqueeze(dim=0),root_anns[1][i].unsqueeze(dim=0),root_anns[2][i].unsqueeze(dim=0)]

                                if not config.detect_with_points.detect_points:
                                    current_count_train_inputs = [[count_train_inputs[0][0][i].unsqueeze(dim=0)],
                                                                  [count_train_inputs[1][0][i].unsqueeze(dim=0), count_train_inputs[1][1][i].unsqueeze(dim=0),
                                                                   count_train_inputs[1][2][i].unsqueeze(dim=0), count_train_inputs[1][3][i].unsqueeze(dim=0),
                                                                   count_train_inputs[1][4][i].unsqueeze(dim=0)]]
                                else:
                                    current_count_train_inputs = [[count_train_inputs[0][0][i].unsqueeze(dim=0)],count_train_inputs[1][i].unsqueeze(dim=0)]

                                #torch.cuda.empty_cache()
                                if self.network_type == "both_for_roots_2":

                                    ######################################################################################
                                    if config.Detect_and_Estimate.use_new_Find:

                                        config.General.binary_model = False
                                        SFMS_lists_len, cls_output_len, current_maps_loss_len = self.find_2_length(current_count_train_inputs)
                                        maps_loss += current_maps_loss_len

                                        SFMS_lists_dia, cls_output_dia, current_maps_loss_dia = self.find_2_diameter(current_count_train_inputs)
                                        maps_loss += current_maps_loss_dia

                                        # the color output is currently binary
                                        config.General.binary_model = True
                                        SFMS_lists_color, cls_output_color, current_maps_loss_color = self.find_2_color(current_count_train_inputs)
                                        maps_loss += current_maps_loss_color

                                        count_input = {"count_input_len": [SFMS_lists_len, cls_output_len, current_roots_anns],
                                                       "count_input_dia": [SFMS_lists_dia, cls_output_dia, current_roots_anns],
                                                       "count_input_color": [SFMS_lists_color, cls_output_color, current_roots_anns]}

                                        ######################################################################################
                                    else:

                                        # the color output is currently binary
                                        config.General.binary_model = True
                                        SFMS_lists, cls_output, current_maps_loss = self.find_2(current_count_train_inputs)
                                        maps_loss += current_maps_loss
                                        count_input = SFMS_lists, cls_output, current_roots_anns #annotations[1][0][0] #corrected_counting_anns

                                if self.network_type == "both":
                                    SFMS_lists, cls_output, current_maps_loss = self.find_2(current_count_train_inputs)
                                    maps_loss += current_maps_loss
                                    count_input = SFMS_lists, cls_output, current_roots_anns  # annotations[1][0][0] #corrected_counting_anns

                                    current_l1 = self.LeanCountingModule(count_input)
                                    l1 += current_l1
                                    #counting_loss = l1 + maps_loss

                                elif self.network_type == "both_for_roots":
                                    SFMS_lists, cls_output, current_maps_loss = self.find_2(current_count_train_inputs)
                                    maps_loss += current_maps_loss
                                    count_input = SFMS_lists, cls_output, current_roots_anns  # annotations[1][0][0] #corrected_counting_anns

                                    #torch.empty_cache()
                                    current_count_loss, current_length_loss, current_diameter_loss = self.LeanModuleForRoots(count_input)

                                    count_loss += current_count_loss
                                    length_loss += current_length_loss
                                    diameter_loss += current_diameter_loss

                                #time.sleep(1)

                                elif self.network_type == "both_for_roots_2":
                                    if config.Detect_and_Estimate.use_new_Find:

                                        # the color output is currently binary
                                        config.General.binary_model = True
                                        config.General.binary_version = "L1Loss"
                                        count_input_color = count_input["count_input_color"][0], count_input["count_input_color"][1], count_input["count_input_color"][2]
                                        current_color_loss = self.LeanCountingModule_color(count_input_color)

                                        config.General.binary_model = False
                                        count_input_len = count_input["count_input_len"][0], count_input["count_input_len"][1], count_input["count_input_len"][2]
                                        count_input_dia = count_input["count_input_dia"][0], count_input["count_input_dia"][1], count_input["count_input_dia"][2]
                                        current_length_loss = self.LeanCountingModule_length(count_input_len)
                                        current_diameter_loss = self.LeanCountingModule_diameter(count_input_dia)

                                    else:
                                        # the color output is currently binary
                                        config.General.binary_model = True
                                        config.General.binary_version = "L1Loss"
                                        current_color_loss = self.LeanCountingModule_color(count_input)

                                        config.General.binary_model = False
                                        current_length_loss = self.LeanCountingModule_length(count_input)
                                        current_diameter_loss = self.LeanCountingModule_diameter(count_input)

                                    color_loss += current_color_loss[0]
                                    length_loss += current_length_loss[0]
                                    diameter_loss += current_diameter_loss[0]


                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':

                            if self.network_type == "both_for_roots_2":
                                length_loss = 0
                                diameter_loss = 0
                                color_loss = 0

                                for i in range(num_of_boxes):
                                    # the color output is currently binary
                                    config.General.binary_model = True
                                    config.General.binary_version = "L1Loss"
                                    current_color_loss = self.CountWithRegModule_color([bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels], corrected_counting_anns[6][i,0].unsqueeze(dim=0).unsqueeze(dim=0)])


                                    config.General.binary_model = False
                                    current_length_loss = self.CountWithRegModule_length([bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels], corrected_counting_anns[6][i,1].unsqueeze(dim=0).unsqueeze(dim=0)])
                                    current_diameter_loss = self.CountWithRegModule_diameter([bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels], corrected_counting_anns[6][i,2].unsqueeze(dim=0).unsqueeze(dim=0)])

                                    color_loss += current_color_loss
                                    length_loss += current_length_loss
                                    diameter_loss += current_diameter_loss

                            else:
                                counting_loss = self.CountWithRegModule([bbox_pyramid_feats[:config.AttributeEstimation.num_of_pyr_levels], corrected_counting_anns[0]])

                        #total_loss = total_loss + counting_loss

                        including_counting = True

                    else:
                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                            if not config.detect_with_points.detect_points:
                                if self.network_type == "both_for_roots_2":
                                    config.General.binary_model = True # for the color model output

                                ######################################################################################
                                if config.Detect_and_Estimate.use_new_Find:
                                    config.General.binary_model = False
                                    SFMS_lists_len, cls_output_len = self.find_2_length(bbox_pyramid_p3)
                                    SFMS_lists_dia, cls_output_dia = self.find_2_diameter(bbox_pyramid_p3)

                                    config.General.binary_model = True
                                    SFMS_lists_color, cls_output_color = self.find_2_color(bbox_pyramid_p3)

                                    count_input={"count_input_len": [SFMS_lists_len, cls_output_len],
                                                 "count_input_dia": [SFMS_lists_dia, cls_output_dia],
                                                 "count_input_color": [SFMS_lists_color, cls_output_color]}

                                    ######################################################################################
                                else:

                                    SFMS_lists, cls_output = self.find_2(bbox_pyramid_p3)
                                    count_input = SFMS_lists, cls_output

                                    if config.General.twoFind_2:
                                        if config.General.twoBackbone_2:
                                            SFMS_lists2, cls_output2 = self.find_2_b(bbox_pyramid_p3_2)

                                        else:
                                            SFMS_lists2, cls_output2 = self.find_2_b(bbox_pyramid_p3)

                                        count_input_2 = SFMS_lists2, cls_output2

                            else:
                                SFMS_lists, cls_output = self.find_2(p3)
                                count_input = SFMS_lists, cls_output

                            if self.network_type == "both":
                                counting_outputs = self.LeanCountingModule(count_input)

                            elif self.network_type == "both_for_roots":
                                counting_outputs = self.LeanModuleForRoots(count_input)

                            elif self.network_type == "both_for_roots_2":

                                if config.Detect_and_Estimate.use_new_Find:

                                    # the color output is currently binary
                                    config.General.binary_model = True
                                    config.General.binary_version = "L1Loss"
                                    count_input_color = count_input["count_input_color"][0], count_input["count_input_color"][1]
                                    color, maps_0_color, maps_1_color, maps_2_color, maps_3_color, maps_4_color, maps_5_color= \
                                        self.LeanCountingModule_color(count_input_color)

                                    config.General.binary_model = False
                                    count_input_len = count_input["count_input_len"][0], count_input["count_input_len"][1]
                                    count_input_dia = count_input["count_input_dia"][0], count_input["count_input_dia"][1]

                                    length, maps_0_len, maps_1_len, maps_2_len, maps_3_len, maps_4_len, maps_5_len = \
                                        self.LeanCountingModule_length(count_input_len)

                                    diameter, maps_0_dia, maps_1_dia, maps_2_dia, maps_3_dia, maps_4_dia, maps_5_dia = \
                                        self.LeanCountingModule_diameter(count_input_dia)

                                    #choose what maps tp pass tp inference - ToDo: pass all and correct eval script accordingly
                                    maps_0, maps_1, maps_2, maps_3, maps_4, maps_5 = maps_0_len, maps_1_len, maps_2_len, maps_3_len, maps_4_len, maps_5_len

                                    counting_outputs = [torch.cat([color, length, diameter], dim=-1), maps_0, maps_1, maps_2, maps_3, maps_4, maps_5]


                                else:

                                    # the color output is currently binary
                                    config.General.binary_model = True
                                    config.General.binary_version = "L1Loss"
                                    color,_ ,_ ,_ ,_ , _, _ = self.LeanCountingModule_color(count_input)

                                    config.General.binary_model = False
                                    if config.General.twoFind_2:
                                        length, maps_0, maps_1, maps_2, maps_3, maps_4, maps_5 = self.LeanCountingModule_length(count_input_2)
                                    else:
                                        length, maps_0, maps_1, maps_2, maps_3, maps_4, maps_5 = self.LeanCountingModule_length(count_input)

                                    diameter,_ ,_ ,_ ,_ , _, _ = self.LeanCountingModule_diameter(count_input)

                                    counting_outputs = [torch.cat([color,length,diameter], dim=-1), maps_0, maps_1, maps_2, maps_3, maps_4, maps_5]

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                            if self.network_type == "both_for_roots_2":
                                counting_outputs = []
                                for i in range(num_of_boxes):
                                    # the color output is currently binary
                                    config.General.binary_model = True
                                    current_color = self.CountWithRegModule_color(bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels])[0]
                                    # if i==0:
                                    #     color = current_color
                                    # else:
                                    #     color = torch.cat((color, current_color), dim=0)

                                    config.General.binary_model = False
                                    current_length = self.CountWithRegModule_length(bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels])[0]
                                    current_diameter= self.CountWithRegModule_diameter(bbox_pyramid_feats[i][:config.AttributeEstimation.num_of_pyr_levels])[0]

                                    counting_outputs.append(torch.cat([current_color,current_length,current_diameter]).unsqueeze(0))
                                    # if i == 0:
                                    #     length = current_length
                                    #     diameter = current_diameter
                                    # else:
                                    #     length = torch.cat((length, current_length), dim=0)
                                    #     diameter = torch.cat((diameter, current_diameter), dim=0)

                                counting_outputs = [torch.cat(counting_outputs, dim=0)]
                                #counting_outputs = [torch.cat([color.unsqueeze(0),length.unsqueeze(0),diameter.unsqueeze(0)], dim=0)]

                            else:
                                counting_outputs = self.CountWithRegModule(bbox_pyramid_feats[:config.AttributeEstimation.num_of_pyr_levels])

                else:
                    sample=None

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
                        bbox_pred_adjusted = torch.empty(0)
                        counting_outputs = None


            if self.training:
                if do_counting and counting_anns is not None:
                    if not including_counting: # don't have bbox predictions
                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                            if self.network_type == "both":
                                return classification_loss, regression_loss, None, None
                            elif self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                                return classification_loss, regression_loss, None, None, None, None

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                            if self.network_type == "both_for_roots_2":
                                return classification_loss, regression_loss, None, None, None
                            else:
                                return classification_loss, regression_loss, None

                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                        if self.network_type == "both":
                            return classification_loss, regression_loss, l1, maps_loss  #counting_loss #total_loss #classification_loss, regression_loss, counting_loss

                        elif self.network_type == "both_for_roots":
                            return classification_loss, regression_loss, count_loss, maps_loss, length_loss, diameter_loss
                        elif self.network_type == "both_for_roots_2":
                            return classification_loss, regression_loss, color_loss, maps_loss, length_loss, diameter_loss

                    elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                        if self.network_type == "both_for_roots_2":
                            return classification_loss, regression_loss, color_loss, length_loss, diameter_loss
                        else:
                            return classification_loss, regression_loss, counting_loss

                else:
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                        if self.network_type == "both":
                            return classification_loss, regression_loss, None, None
                        elif self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                            return classification_loss, regression_loss, None, None, None, None

                    elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                        if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                            return classification_loss, regression_loss, None, None, None
                        else:
                            return classification_loss, regression_loss, None

            else:
                if bbox_pred_adjusted.shape[0] > 0: #len(bbox_crops_list)>0:
                    if not config.detect_with_points.detect_points:
                        return detection_outputs, counting_outputs, sample, relevant_points, crops_orig_boxes #bbox_pred_adjusted,....
                    else:
                        return detection_outputs, counting_outputs, None, relevant_points, crops_orig_boxes,

                else:
                    return detection_outputs, counting_outputs, None, None, None  #[],...


        elif self.network_type == "counting_reg":
            pyramid_feats = self.backbone_1(img_batch)

            if self.training:
                if config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    reg_loss = self.CountWithRegModule(
                        [pyramid_feats[:config.AttributeEstimation.num_of_pyr_levels], annotations[0].cuda()])

                return reg_loss

            else:
                if config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    reg_outputs = self.CountWithRegModule(pyramid_feats[:config.AttributeEstimation.num_of_pyr_levels])

                return reg_outputs




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

            im=torch.tensor(im).float().permute(2,0,1).unsqueeze(dim=0).cuda()

        indices = torch.tensor([0, 1, 2, 3]).cuda()
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
            if self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                if self.training or (not self.training and anns is not None):
                    current_img = bbox_crops[b][0][0].permute(1, 2, 0).cpu()
                else:
                    current_img = bbox_crops[b][0].permute(1, 2, 0).cpu()
            else:
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

                elif self.network_type == "both_for_roots" or self.network_type == "both_for_roots_2":
                    annotations_group_num_of_points = 0
                    annotation_map_1 = torch.empty((80,80), dtype=float)
                    annotation_map_2 = torch.empty((80, 80), dtype=float)
                    annotation_map_3 = torch.empty((80, 80), dtype=float)
                    annotation_map_4 = torch.empty((80, 80), dtype=float)
                    annotation_map_5 = torch.empty((80, 80), dtype=float)

                    sample['annot'] = [[annotations_group_num_of_points], annotation_map_1, annotation_map_2,
                                       annotation_map_3, annotation_map_4, annotation_map_5]


                if self.network_type =="both_for_roots" or self.network_type =="both_for_roots_2":

                    anns_idx = bbox_crops[b][1].item()

                    if anns_idx!=-1:
                        anns_loc = ((anns[1][0][0][:,2] == anns_idx).nonzero(as_tuple=True)[0]).item()
                        roots_anns = anns[1][0][0][anns_loc]  # [int(anns_idx)]

                        points_color = roots_anns[0].unsqueeze(dim=0) #points_count
                        length = roots_anns[3].unsqueeze(dim=0)
                        diameter = roots_anns[4].unsqueeze(dim=0)
                        gt_idx = torch.tensor(anns_idx, dtype=float).unsqueeze(dim=0)
                        sample['roots_annot'] = torch.cat([points_color, length, diameter, gt_idx]) #points_count

                    else:
                        sample['roots_annot'] = torch.tensor([-1, 0, 0, -1], dtype=float) # [color, length, dia, gt_box_id]

                    if config.detect_with_points.detect_points:
                        sample['gt_box_id']=anns_idx
                        sample['gt_box_maps'] = []
                        for i in range(5):
                            sample['gt_box_maps'].append(np.array(anns[2][1][anns_loc][i]))  #


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
                if network_type== "both_for_roots" or network_type == "both_for_roots_2":
                    if p['bbox_id']!=box_id:
                        continue
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

