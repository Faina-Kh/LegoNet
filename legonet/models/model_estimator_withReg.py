import torch.nn as nn

import legos
import config


class ImageEstimatorWithReg(nn.Module): # CountingReg
    """Counting-only regression model built from a backbone and regression head.

    The model composes a shared ResNet-FPN backbone from ``legos`` with the
    regression-based attribute estimator. It preserves the legacy
    ``counting_reg`` training and inference outputs expected by ``runner.py``.
    """

    def __init__(
        self,
        dataset,
        network_type: str,
        num_classes: int,
        pretrained: bool = True,
        resnet_depth: int = 50,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.network_type = network_type
        self.num_classes = num_classes

        self.backbone = legos.ResNetBackboneModule(
            depth=resnet_depth,
            pretrained=pretrained,
            name='backbone_for_attribute',
        )
        self.estimator = legos.RegressionBasedEstimator(num_classes)

    def freeze_detector(self) -> None:
        """No-op for the counting-only model.

        ``runner.py`` calls ``freeze_detector`` for every model type, so the
        counting-only variants provide a compatible stub.
        """

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs[0]
            annotations = None

        pyramid_feats = self.backbone(img_batch)
        pyramid_inputs = pyramid_feats[: config.AttributeEstimation.num_of_pyr_levels]

        if self.training:
            return self.estimator(
                [pyramid_inputs, annotations[0].to(config.General.device)]
            )

        return self.estimator(pyramid_inputs)
