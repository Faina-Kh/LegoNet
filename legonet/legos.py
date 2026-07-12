import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import config
from legonet import modular_losses
import gc
from pathlib import Path


resnet_module_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_WEIGHTS_DIR = REPO_ROOT / ".cache" / "pretrained"
LOCAL_RESNET_WEIGHTS = {
    "resnet50": REPO_ROOT / "resnet50-19c8e357.pth",
}


def load_pretrained_resnet_weights(module_url):
    """Load ResNet weights from the repo first, then a repo-local cache."""
    local_weights_file = LOCAL_RESNET_WEIGHTS.get(module_url)
    if local_weights_file is not None and local_weights_file.exists():
        return torch.load(str(local_weights_file), map_location="cpu")

    PRETRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return model_zoo.load_url(
        resnet_module_urls[module_url],
        model_dir=str(PRETRAINED_WEIGHTS_DIR),
    )


def init_module_weights(m):
    for m in m.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class LocalNMS(nn.Module):
    '''
    This layer computes a local Non-Maxima Suppression keeping only a single non-zero element for a local window
    input:
        kernel_size - size of window for local NMS.
        strides - overlap factor
        beta - to decay the rest of the element an exponent is used and 'beta' is the exponent magnitude element
    '''
    def __init__(self, kernel_size = (3,3), strides=(1,1), beta=100, name = "LocalNMS"):
        super(LocalNMS, self).__init__()
        self.name = name
        self.kernel_size = kernel_size
        self.strides = strides
        self.beta = beta
        self.max_pool_layer = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.strides, padding=1)

    def forward(self, x):
        '''
        :param inputs: a 2-dim matrix
        :return: the local NMS of inputs
        '''
        p = x
        q = self.max_pool_layer(p)
        abs_p_minus_q= torch.abs(p-q)
        exp_abs = torch.exp(-abs_p_minus_q*self.beta)
        p_hat = torch.mul(p,exp_abs)
        return p_hat


class GlobalSumPooling2D(nn.Module):
    '''
    This layer computes the sum of elements in a 2-dim matrix
    '''
    def __init__(self):
        super(GlobalSumPooling2D, self).__init__()

        self.name = "GlobalSumPooling2D"

    def forward(self, x):
        '''
        :param x: a 2-dim matrix
        :return: the global sum of x
        '''

        sum_pooling = torch.sum(x, dim=(1,2)).view(x.size(0),1) #torch.sum(x.view(x.size(0), -1))
        return sum_pooling


class SmoothStepFunction(nn.Module):
    def __init__(self, threshold = 0.8, beta = 15, name = "SmoothStepFunction"):
        super(SmoothStepFunction,self).__init__()
        self.name = name
        self.threshold = threshold
        self.beta = beta

    def forward(self, x):
        threshold_factor = torch.ones_like(x) * self.threshold
        sigmoid_input = torch.sub(x, threshold_factor) * self.beta
        smooth_step_function = torch.sigmoid(sigmoid_input)
        return smooth_step_function



########################################################################################################################
# Backbone related modules
########################################################################################################################

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class ResNetBackboneModule(nn.Module):

    def __init__(self, depth=50, pretrained=False, name = "Backbone_Module_Res"):

        self.name = name
        self.pretrained = pretrained

        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
            module_url = 'resnet18'
        elif depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
            module_url = 'resnet34'
        elif depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            module_url = 'resnet50'
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
            module_url = 'resnet101'
        elif depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
            module_url = 'resnet152'
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        self.inplanes = 64
        super(ResNetBackboneModule, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        if not self.pretrained:
            init_module_weights(self)
        else:
            self.load_state_dict(load_pretrained_resnet_weights(module_url), strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        return features

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

########################################################################################################################
# 'Find' related modules
########################################################################################################################

class FeatureClassification(nn.Module):
    def __init__(self, num_features_in = 256, num_classes=80, num_anchors=9, feature_size=256, name = "FeatureClassification", task=''): #pretrained=False,
        super(FeatureClassification, self).__init__()

        self.task = task
        self.name = name

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv1.weight.data.normal_(mean=0.0, std=0.01)
        self.conv1.bias.data.zero_()

        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2.weight.data.normal_(mean=0.0, std=0.01)
        self.conv2.bias.data.zero_()

        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3.weight.data.normal_(mean=0.0, std=0.01)
        self.conv3.bias.data.zero_()

        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4.weight.data.normal_(mean=0.0, std=0.01)
        self.conv4.bias.data.zero_()

        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output.weight.data.normal_(mean=0.0, std=0.01)
        self.output.bias.data.zero_()

        self.output_act = nn.Sigmoid()

        if self.task == 'attribute_estimation':

            self.mid_conv1 = nn.Conv2d(feature_size, num_classes, kernel_size=1)
            self.mid_conv1.weight.data.normal_(mean=0.0, std=0.01)
            self.mid_conv1.bias.data.zero_()

            self.mid_act1 = nn.ReLU()

            self.mid_conv2 = nn.Conv2d(feature_size, num_classes, kernel_size=1)
            self.mid_conv2.weight.data.normal_(mean=0.0, std=0.01)
            self.mid_conv2.bias.data.zero_()

            self.mid_act2 = nn.ReLU()

            self.mid_conv3 = nn.Conv2d(feature_size, num_classes, kernel_size=1)
            self.mid_conv3.weight.data.normal_(mean=0.0, std=0.01)
            self.mid_conv3.bias.data.zero_()

            self.mid_act3 = nn.ReLU()

            self.mid_conv4 = nn.Conv2d(feature_size, num_classes, kernel_size=1)
            self.mid_conv4.weight.data.normal_(mean=0.0, std=0.01)
            self.mid_conv4.bias.data.zero_()

            self.mid_act4 = nn.ReLU()

            self.output_counting = nn.Conv2d(feature_size, num_classes, kernel_size=3, padding=1)
            self.output_counting.weight.data.normal_(mean=0.0, std=0.01)
            self.output_counting.bias.data.fill_(0.1)  #.zero_()

            self.output_act_counting= nn.ReLU()

    def forward(self, x):

        x1 = x.permute(0, 2, 3, 1)

        if self.task == 'bbox_detection':
            out = self.conv1(x)
            out = self.act1(out)

            out = self.conv2(out)
            out = self.act2(out)

            out = self.conv3(out)
            out = self.act3(out)

            out = self.conv4(out)
            out = self.act4(out)

            out = self.output(out)
            out = self.output_act(out)

            # out is B x C x W x H, with C = n_classes + n_anchors
            out1 = out.permute(0, 2, 3, 1)

            return {'feature_maps': x1, 'find_maps': out1}

        elif self.task == 'attribute_estimation': #'counting': #plt.imsave('vis' + '/' + 'out.png', out) #print(torch.min(out.data)) print(torch.max(out.data))
            outputs = []
            out = self.conv1(x)
            #print('out', torch.min(out.data), torch.max(out.data))
            out = self.act1(out)

            mid_out1 = self.mid_conv1(out)
            #print('mid1', torch.min(mid_out1.data), torch.max(mid_out1.data))
            mid_out1 = self.mid_act1(mid_out1)
            outputs.append(mid_out1)
            #print('mid1', torch.min(mid_out1.data), torch.max(mid_out1.data))

            out = self.conv2(out)
            #print('out', torch.min(out.data), torch.max(out.data))
            out = self.act2(out)

            mid_out2 = self.mid_conv2(out)
            #print('mid2', torch.min(mid_out2.data), torch.max(mid_out2.data))
            mid_out2 = self.mid_act2(mid_out2)
            outputs.append(mid_out2)
            #print('mid2', torch.min(mid_out2.data), torch.max(mid_out2.data))

            out = self.conv3(out)
            #print('out', torch.min(out.data), torch.max(out.data))
            out = self.act3(out)

            mid_out3 = self.mid_conv3(out)
            mid_out3 = self.mid_act3(mid_out3)
            #print('mid3', torch.min(mid_out3.data), torch.max(mid_out3.data))
            outputs.append(mid_out3)
            #print('mid3', torch.min(mid_out3.data), torch.max(mid_out3.data))

            out = self.conv4(out)
            #print('out', torch.min(out.data), torch.max(out.data))
            out = self.act4(out)

            mid_out4 = self.mid_conv4(out)
            #print('mid4', torch.min(mid_out4.data), torch.max(mid_out4.data))
            mid_out4 = self.mid_act4(mid_out4)
            #print('mid4', torch.min(mid_out4.data), torch.max(mid_out4.data))
            outputs.append(mid_out4)

            outputs.append(out)

            out = self.output_counting(out)
            ###############################################################
            w = self.output_counting.weight.data
            b = self.output_counting.bias.data
            #print('output_counting - w, min:{:.6f}, max:{:.6f}'.format(torch.min(w), torch.max(w)))
            #print('output_counting - b,  min:{:.6f}, max:{:.6f}'.format(torch.min(b), torch.max(b)))

            #print('out - min:{:.5f}, max:{:.5f}'.format(torch.min(out.data), torch.max(out.data)))
            ###############################################################

            out = self.output_act_counting(out)
            #print('out relu- min:{:.5f}, max:{:.5f}'.format(torch.min(out.data), torch.max(out.data)))
            outputs.append(out)

            # print('output_counting - w, min:{:.6f}, max:{:.6f}'.format(torch.min(w), torch.max(w)))
            # print('output_counting - b,  min:{:.6f}, max:{:.6f}\n'.format(torch.min(b), torch.max(b)))
            #
            # print('mid4 - w, min:{:.6f}, max:{:.6f}'.format(torch.min(self.mid_conv4.weight.data), torch.max(self.mid_conv4.weight.data)))
            # print('mid4 - b,  min:{:.6f}, max:{:.6f}\n'.format(torch.min(self.mid_conv4.bias.data), torch.max(self.mid_conv4.bias.data)))
            #
            # print('mid3 - w, min:{:.6f}, max:{:.6f}'.format(torch.min(self.mid_conv3.weight.data), torch.max(self.mid_conv3.weight.data)))
            # print('mid3 - b,  min:{:.6f}, max:{:.6f}\n'.format(torch.min(self.mid_conv3.bias.data), torch.max(self.mid_conv3.bias.data)))
            #
            # print('mid2 - w, min:{:.6f}, max:{:.6f}'.format(torch.min(self.mid_conv2.weight.data), torch.max(self.mid_conv2.weight.data)))
            # print('mid2 - b,  min:{:.6f}, max:{:.6f}\n'.format(torch.min(self.mid_conv2.bias.data), torch.max(self.mid_conv2.bias.data)))
            #
            # print('mid1 - w, min:{:.6f}, max:{:.6f}'.format(torch.min(self.mid_conv1.weight.data), torch.max(self.mid_conv1.weight.data)))
            # print('mid1 - b,  min:{:.6f}, max:{:.6f}\n'.format(torch.min(self.mid_conv1.bias.data), torch.max(self.mid_conv1.bias.data)))


            new_outputs = []
            for output in outputs:
                new_outputs.append(output.permute(0, 2, 3, 1))

            new_outputs[0] = new_outputs[0].squeeze(3)
            new_outputs[1] = new_outputs[1].squeeze(3)
            new_outputs[2] = new_outputs[2].squeeze(3)
            new_outputs[3] = new_outputs[3].squeeze(3)
            new_outputs[5] = new_outputs[5].squeeze(3)

            return {'feature_maps': x1, 'find_maps': new_outputs}


class FindModule(nn.Module):
    def __init__(self, num_features_in = 256, num_classes=80, pretrained=False, num_anchors = 9, feature_size=256,
                 name = "Find_Module", task="", Find_for_count = ""):
        super(FindModule, self).__init__()

        self.Find_for_count = Find_for_count
        self.task = task
        self.name = name
        self.pretrained = pretrained

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.feature_classification = FeatureClassification(num_features_in=num_features_in, num_classes=num_classes,
                                                            feature_size= feature_size, task=task)

        self.SmoothStepFunction = SmoothStepFunction(threshold=0.4, beta=1)
        self.LocalNMS = LocalNMS(kernel_size=(3, 3), strides=(1, 1), beta=100)
        self.SmoothStepFunction1 = SmoothStepFunction(threshold=0.8, beta=15)
        self.GlobalSumPooling2D = GlobalSumPooling2D()

        if self.task == 'bbox_detection':
            self.focalLoss = modular_losses.FindFocalLoss()

        elif self.task == 'attribute_estimation':
            self.focalLoss = modular_losses.FindKeyPointsFocalLoss()

        if self.task == 'attribute_estimation':
            if not self.Find_for_count: #if config.General.binary_model: #if config.General.with_new_layers:
                self.sigmoid_for_binary = torch.nn.Sigmoid()
                self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size=3, padding=1)
                self.conv1.weight.data.normal_(mean=0.0, std=0.01)
                self.conv1.bias.data.zero_()
                self.conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
                self.conv2.weight.data.normal_(mean=0.0, std=0.01)
                self.conv2.bias.data.zero_()
                self.act_1 = nn.ReLU()
                self.linear = torch.nn.Linear(6400,1)
                self.sigmoid_for_binary_2 = torch.nn.Sigmoid()


    def forward(self, inputs):

        if self.training:

            if self.task=='bbox_detection':
                pyramid_feats, anchors, annotations = inputs

            elif self.task == 'attribute_estimation':
                pyramid_feats, annotations = inputs

        else:
            pyramid_feats = inputs

        #output_maps are B x  W x H x C

        gc.collect()
        torch.cuda.empty_cache()

        SFMS_lists = [self.feature_classification(feature) for feature in pyramid_feats]

        if self.task == 'bbox_detection':
            if self.training: # and not self.freeze_detection:
                classifications = []
                for SFMS in SFMS_lists:
                    batch_size, width, height, channels = SFMS['find_maps'].shape
                    out2 = SFMS['find_maps'].view(batch_size, width, height, self.num_anchors, self.num_classes)
                    classifications.append(out2.contiguous().view(batch_size, -1, self.num_classes))

                classifications = torch.cat(classifications, dim=1)
                find_loss = self.focalLoss(classifications, anchors, annotations)#modular_losses.FindFocalLoss()(classifications, anchors, annotations)
                return SFMS_lists, find_loss

            else:
                return SFMS_lists

        elif self.task=='attribute_estimation':

            cls_output = []

            for SFMS in SFMS_lists:

                classification = SFMS['find_maps']

                classification_output = classification[-1]
                cls_output_Step_Function1 = self.SmoothStepFunction(classification_output)
                cls_output_MaxPooled = self.LocalNMS(cls_output_Step_Function1)
                cls_output_Step_Function2 = self.SmoothStepFunction1(cls_output_MaxPooled)

                if not self.Find_for_count: #(config.General.binary_model and not config.prev_color_model) or config.General.other
                    x1 = cls_output_Step_Function2.unsqueeze(dim=1)
                    x1 = self.conv1(x1)
                    x1 = self.act_1(x1)
                    x1 = self.conv2(x1)
                    x1 = self.sigmoid_for_binary(x1)
                    x1 = x1.view(x1.shape[0],6400)
                    x1 = self.linear(x1)
                    cls_output_downsampled = self.sigmoid_for_binary_2(x1)

                else:
                    cls_output_downsampled = self.GlobalSumPooling2D(cls_output_Step_Function2)

                cls_output.append(cls_output_downsampled)

                if self.training:
                    if config.AttributeEstimation.inter_losses:
                        l1, l2,l3,l4,l5 = [self.focalLoss(annotations[0], classification[0]),
                                            self.focalLoss(annotations[1], classification[1]),
                                            self.focalLoss(annotations[2], classification[2]),
                                            self.focalLoss(annotations[3], classification[3]),
                                            self.focalLoss(annotations[4], classification[5])]

                        maps_loss = l1+ l2+l3+l4+l5
                    else:
                        maps_loss = self.focalLoss(annotations[4], classification[5])


            if self.training:
                return SFMS_lists, cls_output, maps_loss

            else:
                return SFMS_lists, cls_output


########################################################################################################################
# 'Where' related modules
########################################################################################################################

class FeatureWhere(nn.Module):
    def __init__(self, num_features_in = 256, pretrained = False, num_anchors=9, feature_size=256, name = "FeatureWhere"):
        super(FeatureWhere, self).__init__()

        self.name = name

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

        if not pretrained:
            init_module_weights(self)


    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class WhereModule(nn.Module):
    def __init__(self, network_type, pretrained = False, num_anchors=9, feature_size=256, name = "Where_Module"): #num_features_in,
        super(WhereModule, self).__init__()

        self.name = name
        self.pretrained = pretrained
        self.num_anchors = num_anchors
        self.network_type = network_type

        self.feature_regression = FeatureWhere(feature_size, pretrained) #num_features_in


    def forward(self, inputs):
        if self.training: #len(inputs) == 3
            # bbox detection module is in training
            pyramid_feats, anchors, annotations = inputs
            #freeze_detection = False
        else:
            pyramid_feats = inputs
            #freeze_detection = True

        regression = torch.cat([self.feature_regression(feature) for feature in pyramid_feats], dim=1)

        if self.training: # and not freeze_detection:
            regression_loss = modular_losses.WhereLoss()(regression, anchors, annotations )
            if self.network_type == "per_object_counting" or self.network_type == "per_object_attributes":
                return regression, regression_loss
            else:
                return regression_loss

        else:
            return regression


class BBoxTransform(nn.Module):

    def __init__(self, name ='BBoxTransform',  mean=None, std=None):
        super(BBoxTransform, self).__init__()

        self.name = name

        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).to(config.General.device)
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).to(config.General.device)
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

########################################################################################################################
# Attribute estimation related modules
########################################################################################################################

class KeypointBasedEstimator(nn.Module):

    def __init__(self, output_size = 1, name = "D+R_module", attribute_name = "", binary_model = False, binary_loss_version = "L1Loss"):
        super(KeypointBasedEstimator, self).__init__() #LeanCountingModule

        self.name = name
        self.attribute_name = attribute_name
        self.binary_model = binary_model
        self.binary_loss_version = binary_loss_version

        # self.ConvForCount = ConvForCount()
        self.GlobalAveragePooling2D = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))  # assumes input size (N, C, H, W), output is (N,C,H_out,W_out)
        # self.SmoothStepFunction = SmoothStepFunction(threshold=0.4, beta=1)
        # self.LocalNMS = LocalNMS(kernel_size=(3, 3), strides=(1, 1), beta=100)
        # self.SmoothStepFunction1 = SmoothStepFunction(threshold=0.8, beta=15)
        # self.GlobalSumPooling2D = GlobalSumPooling2D()

        #self.last_reg_layer = torch.nn.Linear(in_features=5, out_features= output_size, bias=True)  # in_features=reg_output_downsampled.size(-1)
        #self.last_reg_layer.weight.data.normal_(mean=0.5, std=0.1)

        self.output_size = output_size

        # self.sec_reg_layer = torch.nn.Linear(in_features=257, out_features=output_size,
        #                                      bias=True)  # in_features=reg_output_downsampled.size(-1)
        # self.sec_reg_layer.weight.data.normal_(mean=0.5, std=0.1)

        self.sec_reg_layer = torch.nn.Linear(in_features=257, out_features=1, bias=True)
        self.sec_reg_layer.weight.data.normal_(mean=0.5, std=0.1)

        self.act_1 = nn.ReLU()

        if output_size > 1:
            self.reg_layer_2 = torch.nn.Linear(in_features=257, out_features = 1, bias=True)
            self.reg_layer_2.weight.data.normal_(mean=0.5, std=0.1)

            self.act_2 = nn.ReLU()

            self.reg_layer_3 = torch.nn.Linear(in_features=257, out_features=1, bias=True)
            self.reg_layer_3.weight.data.normal_(mean=0.5, std=0.1)

            self.act_3 = nn.ReLU()

            self.reg_layer_4 = torch.nn.Linear(in_features=257, out_features=1, bias=True)
            self.reg_layer_4.weight.data.normal_(mean=0.5, std=0.1)

            self.act_4 = nn.ReLU()

        self.L1loss = nn.L1Loss()

        if self.binary_model: #config.General.binary_model: # or config.General.with_new_layers_for_both:
            self.sigmoid_for_binary1 = torch.nn.Sigmoid()
            self.sigmoid_for_binary2 = torch.nn.Sigmoid()

            self.forBinary_reg_layer_L1 = torch.nn.Linear(in_features=257, out_features=1, bias=True)
            self.forBinary_reg_layer_crossEnt = torch.nn.Linear(in_features=257, out_features=2, bias=True)
            self.crossEnt = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.67, 0.33]))
            self.forBinary_reg_layer_L1.weight.data.normal_(mean=0.01, std=0.01)
            self.forBinary_reg_layer_crossEnt.weight.data.normal_(mean=0.01, std=0.01)
            self.binaryLoss = torch.nn.BCELoss()
        #if loss_for == "color":
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
            self.conv1.weight.data.normal_(mean=0.0, std=0.01)
            self.conv1.bias.data.zero_()

    def forward(self, inputs):

        if self.training:
            #classification_SFMS, annotations = inputs
            SFMS_list, cls_output,  annotations = inputs
        else:
            #classification_SFMS = inputs
            SFMS_list, cls_output = inputs


        out1 = []
        classification_maps = [[],[],[],[],[],[]]

        for i in range(len(SFMS_list)):
            cls_output_downsampled = cls_output[i]

            classification = SFMS_list[i]['find_maps']
            classification_last_map_for_reg = classification[-2]

            if self.binary_model: #config.General.binary_model:
                # This makes sure, the 256-feature vector will not have only positive values
                classification_last_map_for_reg = self.conv1(classification_last_map_for_reg.permute(0,3,1,2))
                classification_last_map_for_reg = self.sigmoid_for_binary1(classification_last_map_for_reg).permute(0,2,3,1)

            classification_last_map_for_reg = self.GlobalAveragePooling2D(classification_last_map_for_reg.permute(0, 3, 1, 2))
            classification_last_map_for_reg = classification_last_map_for_reg.squeeze(dim=3).squeeze(dim=2)

            reg_output_downsampled = torch.cat([classification_last_map_for_reg, cls_output_downsampled],dim=-1)

            if self.binary_model: #config.General.binary_model:
                #reg_output_downsampled = self.sigmoid_for_binary1(reg_output_downsampled)

                if self.binary_loss_version == "L1Loss":
                    reg_output_downsampled = self.forBinary_reg_layer_L1(reg_output_downsampled)
                    sec_reg_output = self.sigmoid_for_binary2(reg_output_downsampled)

                elif self.binary_loss_version == "crossEnt":
                    reg_output_downsampled = self.forBinary_reg_layer_crossEnt(reg_output_downsampled)
                    sec_reg_output = reg_output_downsampled.softmax(dim=1)
            else:
                sec_reg_output = self.sec_reg_layer(reg_output_downsampled)
                sec_reg_output = self.act_1(sec_reg_output)

            out1.append(sec_reg_output)

            if self.output_size > 1:

                out2 = self.reg_layer_2(reg_output_downsampled)
                out2 = self.act_2(out2)
                out1.append(out2)

                out3 = self.reg_layer_3(reg_output_downsampled)
                out3 = self.act_3(out3)
                out1.append(out3)

                out4 = self.reg_layer_4(reg_output_downsampled)
                out4 = self.act_4(out4)
                out1.append(out4)

                out1 = torch.cat([out1[0], out1[1], out1[2], out1[3]], dim=1)

            # if in the future it will run on multiple scales of pi
            for j in range(len(classification)):
                classification_maps[j].append(classification[j])

        if len(SFMS_list) >1:
            out1 = torch.cat(out1, dim=-1)
            #output = self.last_reg_layer(out1)
            if self.output_size == 1:
                output = torch.mean(out1, dim=1) #

        else:
            if self.output_size == 1:
                output = out1[0]
            else:
                output = out1

        for j in range(len(classification_maps)):
            classification_maps[j]=torch.cat(classification_maps[j], dim=0)

        if self.training:
            losses=[]
            for i in range(self.output_size):
                if self.output_size > 1:
                    losses.append(self.L1loss(annotations[0].squeeze(dim=-1)[:,i], output[:,i]))
                else:
                    if self.binary_model: #config.General.binary_model:
                        if self.binary_loss_version == "L1Loss":
                            losses.append(self.L1loss(annotations[0].float(), output[:,0]))
                        elif self.binary_loss_version == "crossEnt":
                            target = annotations[0].squeeze(dim=1).to(torch.long).to(config.General.device)  #
                            losses.append(self.crossEnt(output, target))
                        #self.binaryLoss(annotations[0].squeeze(dim=1), output[:, i].detach()))
                    else:
                        if self.attribute_name == "length":
                            losses.append(self.L1loss(annotations[1].float(), output[:, 0])) # self.L1loss(annotations[0].squeeze(dim=1), output[:, i]))
                        elif self.attribute_name == "diameter":
                            losses.append(self.L1loss(annotations[2].float(), output[:, 0]))
                        elif self.attribute_name == "color":
                            losses.append(self.L1loss(annotations[0].float(), output[:, 0])) #self.L1loss(annotations[0].squeeze(dim=1), output[:, i]))
                        else:
                            losses.append(self.L1loss(annotations[0].float(), output))


            return (losses)

        else:
            if self.binary_model and self.binary_loss_version == "crossEnt": #config.General.binary_model
                output = torch.argmax(output)

            if self.binary_model and self.binary_loss_version == "L1Loss": #config.General.binary_model
                output = torch.round(output)

            return [output,
                    classification_maps[0], classification_maps[1], classification_maps[2], classification_maps[3], classification_maps[4],classification_maps[5]]


class EstimateRegSubmodel(nn.Module):

    def __init__(self, num_features_in=256, FC_num_of_neurons = 128, feature_size = 256, name = 'FC_submodel',
                 binary_model = False):
        super(EstimateRegSubmodel, self).__init__()

        self.name = name
        self.binary_model = binary_model

        self.FC_num_of_neurons = FC_num_of_neurons

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1) #'strides': 1,'padding': 'same'
        self.conv1.weight.data.normal_(mean=0.0, std=0.01)
        self.conv1.bias.data.zero_()
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)  # 'strides': 1,'padding': 'same'
        self.conv2.weight.data.normal_(mean=0.0, std=0.01)
        self.conv2.bias.data.zero_()
        self.act2 = nn.ReLU()

        self.GlobalAveragePooling2D = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))  # assumes input size (N, C, H, W), output is (N,C,H_out,W_out)

        self.FC_regression = torch.nn.Linear(in_features=256, out_features=FC_num_of_neurons, bias=True)
        # torch.nn.init.xavier_uniform_(self.FC_regression)  #Glorot init, the default in keras Dense
        self.FC_regression.bias.data.zero_()

        self.act_Dense1 = nn.ReLU()

        self.FC_regression2 = torch.nn.Linear(in_features=FC_num_of_neurons, out_features=FC_num_of_neurons//2, bias=True)
        #torch.nn.init.xavier_uniform_(self.FC_regression2) #Glorot init, the default in keras Dense
        self.FC_regression2.bias.data.zero_()
        self.act_Dense2 = nn.ReLU()

        self.regression_output = torch.nn.Linear(in_features=FC_num_of_neurons//2, out_features=2, bias=True)
        #torch.nn.init.xavier_uniform_(self.regression_output)  # Glorot init, the default in keras Dense
        self.regression_output.bias.data.zero_()

        if self.binary_model or config.Detect_and_Estimate.type == "per_object_attributes":
            self.act_Dense1_sig = nn.Sigmoid()
            self.act_Dense2_sig = nn.Sigmoid()
            self.act_Dense3 = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        GlobalAvgPool_features = self.GlobalAveragePooling2D(out) # needs to be 256*1 vector
        FC_regression = self.FC_regression(GlobalAvgPool_features.squeeze(dim=3).squeeze(dim=2))

        if self.binary_model:
            FC_regression = self.act_Dense1_sig(FC_regression)
        else:
            FC_regression = self.act_Dense1(FC_regression)

        FC2_regression = self.FC_regression2(FC_regression)

        if self.binary_model:
            FC2_regression = self.act_Dense2_sig(FC2_regression)
        else:
            FC2_regression = self.act_Dense2(FC2_regression)

        regression_output = self.regression_output(FC2_regression).double() # activation='relu'

        if self.binary_model:
            regression_output[:,0] = self.act_Dense3(regression_output[:,0])


        return regression_output


class RegressionBasedEstimator(nn.Module):

    def __init__(self, num_classes, name = "MSR", binary_model = False):
        super(RegressionBasedEstimator, self).__init__()

        self.name = name
        self.num_classes = num_classes
        self.binary_model = binary_model

        self.L1loss = nn.L1Loss()  # the mae loss
        self.mseLoss = nn.MSELoss()
        self.estimationLoss = modular_losses.BnnLoss()

        self.regSubmodel = EstimateRegSubmodel(binary_model = self.binary_model)

    def forward(self, inputs):

        if self.training:
            pyramid_feats, annotations = inputs
        else:
            pyramid_feats = inputs

        output = [self.regSubmodel(feature) for feature in pyramid_feats]

        best_out = []
        for i in range(pyramid_feats[0].shape[0]): # num of input images
            best_p=-1
            min_sig=1000000
            for p in range(len(pyramid_feats)):
                # find min sig per image
                if output[p][i][1] < min_sig:
                    min_sig = output[p][i][1]
                    best_p = p

            best_out.append(output[best_p][i,:].unsqueeze(dim=0))

        best_out=torch.cat(best_out, dim=0)


        if self.training:
            if config.Detect_and_Estimate.type == "per_object_attributes":
                return (self.estimationLoss(annotations.squeeze(dim=1), best_out))
            return (self.estimationLoss(annotations.squeeze(dim=1), best_out.float()))

        else:
            if self.binary_model:
                return [torch.round(best_out[:, 0])]
            return [best_out[:,0]]

########################################################################################################################

