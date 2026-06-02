import torch.nn as nn

from legonet import legos
import config


class ImageEstimatorWithKeypoints(nn.Module):
    """Per-Image attribute estimation model based on keypoints detection, built from a backbone, finder, and head.

    The model composes the shared backbone with the attribute-estimation finder
    operating on the P3 feature map, then feeds those maps into the keyPoints based
    estimation head.
    """

    def __init__(
        self,
        dataset,
        network_type: str,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.network_type = network_type
        self.num_classes = num_classes

        self.backbone = legos.ResNetBackboneModule(name='backbone_for_attribute')
        self.find = legos.FindModule(
            num_classes=num_classes,
            name='find_for_attribute',
            task='attribute_estimation',
            Find_for_count=True,
        )
        self.estimator = legos.KeypointBasedEstimator() #self.LeanCountingModule

        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    # def freeze_detector(self) -> None:
    #     """No-op for the counting-only model.
    #
    #     ``runner.py`` calls ``freeze_detector`` for every model type, so the
    #     counting-only variants provide a compatible stub.
    #     """

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs[0]
            annotations = None

        pyramid_feats = self.backbone(img_batch)
        p3 = pyramid_feats[0]

        if self.training:
            annotations_on_device = [
                annotation.to(config.General.device) for annotation in annotations
            ]
            count_train_inputs = [[p3], annotations_on_device[1:6]]
            sfms_lists, cls_output, maps_loss = self.find(count_train_inputs)
            count_input = sfms_lists, cls_output, annotations_on_device
            l1_loss = self.estimator(count_input)[0] #[0])
            return l1_loss, maps_loss

        sfms_lists, cls_output = self.find([p3])
        return self.estimator((sfms_lists, cls_output))
