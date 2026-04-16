import torch
import torch.nn as nn
import torchvision.ops
import math

from legacy_scripts import losses
import anchors
from legos import legos_3
import config


class BBOX_Detection(nn.Module):

    def __init__(self, network_type = 'bbox_detection', num_classes = 1, ResNet_depth = 50,
                 pretrained = True, num_anchors = 9, min_score = config.Detection.min_score, prior = 0.01,
                 freeze_detection = False):
        super(BBOX_Detection, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.min_score = min_score
        self.freeze_detection = freeze_detection

        self.backbone_1 = legos_3.ResNetBackboneModule(depth=ResNet_depth, pretrained=pretrained, name='backbone_for_detect')
        self.find_1 = legos_3.FindModule(num_classes=num_classes, name='find_for_detect', task='detection') #,
                                         #freeze_detection=self.freeze_detection) #num_features_in = in_channels,
        self.where = legos_3.WhereModule(network_type) #, num_features_in = in_channels

        # weights initialization
        self.find_1.feature_classification.output.weight.data.fill_(0)
        self.find_1.feature_classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.where.feature_regression.output.weight.data.fill_(0)
        self.where.feature_regression.output.bias.data.fill_(0)

        if config.Detection.change_anchors:
            self.anchors = anchors.Anchors(ratios=config.Detection.ratios)
        else:
            self.anchors = anchors.Anchors()

        self.regressBoxes = legos_3.BBoxTransform()
        self.clipBoxes = legos_3.ClipBoxes()
        self.focalLoss = losses.FocalLoss()

    def freeze_detector(self):
        self.eval()
        # for p in self.parameters():
        #     p.requires_grad = False

        # freeze gradients of detection
        for param in self.backbone_1.parameters():
            param.requires_grad = False

        for param in self.find_1.parameters():
            param.requires_grad = False

        for param in self.where.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        if self.training: # and not self.freeze_detection:
            img_batch, annotations = inputs
        else:
            img_batch = inputs[0]

        anchors = self.anchors(img_batch)
        pyramid_feats = self.backbone_1(img_batch)

        if self.training: # and not self.freeze_detection:
            detect_train_inputs = pyramid_feats, anchors, annotations
            classification_SFMS, classification_loss = self.find_1(detect_train_inputs)
            regression_loss = self.where(detect_train_inputs)
            return classification_loss, regression_loss

        else:
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

