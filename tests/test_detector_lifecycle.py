import unittest

import torch.nn as nn

from legonet.models.detector_lifecycle import DetectorLifecycleMixin


class _Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_1 = nn.Linear(2, 2)
        self.find_1 = nn.Linear(2, 2)
        self.where = nn.Linear(2, 2)


class _Model(DetectorLifecycleMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(1)
        self.bbox_detection = _Detector()


class DetectorLifecycleTests(unittest.TestCase):
    def test_freeze_bn_sets_batch_norm_to_evaluation_mode(self):
        model = _Model()

        model.freeze_bn()

        self.assertFalse(model.batch_norm.training)

    def test_freeze_detector_disables_detector_training_and_gradients(self):
        model = _Model()

        model.freeze_detector()

        self.assertFalse(model.bbox_detection.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.bbox_detection.parameters())
        )


if __name__ == "__main__":
    unittest.main()
