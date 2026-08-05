"""Tests for object-detection metric calculations."""

import unittest
from unittest import mock

import numpy as np

from legonet.eval import detection_eval


class _Generator:
    """Minimal dataset interface used by the detection evaluator."""

    def __init__(self, num_classes: int, num_images: int):
        self._num_classes = num_classes
        self._num_images = num_images

    def __len__(self) -> int:
        return self._num_images

    def num_classes(self) -> int:
        return self._num_classes


class DetectionEvaluationTests(unittest.TestCase):
    """Characterize detection precision/recall semantics."""

    def test_empty_gt_scope_counts_images_not_empty_class_slots(self):
        """An image is empty only when every class has no annotations."""
        annotations = [
            [np.zeros((0, 4)), np.zeros((0, 4))],
            [np.array([[0.0, 0.0, 1.0, 1.0]]), np.zeros((0, 4))],
        ]

        self.assertEqual(
            detection_eval._count_empty_gt_images(annotations),
            1,
        )

    def test_combined_model_detects_images_without_point_annotations(self):
        """Roots empty images still pass through the bbox detector."""
        generator = _Generator(num_classes=1, num_images=1)
        sampler = mock.Mock(groups=[[0]])
        image = mock.Mock()
        processed_image = mock.Mock()
        image.clone.return_value.detach.return_value = processed_image
        processed_image.to.return_value.float.return_value = "image"
        data = {
            "img": image,
            "bbox_annot": mock.Mock(),
            "scale": np.array([1.0]),
        }
        scores = mock.Mock()
        labels = mock.Mock()
        boxes = mock.Mock()
        scores.cpu.return_value.numpy.return_value = np.array([0.9])
        labels.cpu.return_value.numpy.return_value = np.array([0])
        boxes.cpu.return_value.numpy.return_value = np.array(
            [[1.0, 2.0, 3.0, 4.0]]
        )
        model = mock.Mock()
        model.bbox_detection.return_value = scores, labels, boxes
        original_network_type = config.General.NETWORK_TYPE
        original_model_type = config.Detect_and_Estimate.type

        try:
            config.General.NETWORK_TYPE = (
                config.NetworkType.detection_and_estimation
            )
            config.Detect_and_Estimate.type = "per_object_attributes"
            detections = detection_eval._get_detections(
                generator,
                model,
                [data],
                sampler,
                score_threshold=0.5,
            )
        finally:
            config.General.NETWORK_TYPE = original_network_type
            config.Detect_and_Estimate.type = original_model_type

        model.bbox_detection.assert_called_once_with(["image"])
        self.assertEqual(detections[0][0].shape, (1, 5))
        self.assertAlmostEqual(detections[0][0][0, 4], 0.9)

    def test_empty_image_detection_counts_as_false_positive(self):
        """A prediction on an empty image lowers full-dataset precision."""
        generator = _Generator(num_classes=1, num_images=2)
        annotations = [
            [np.array([[0.0, 0.0, 10.0, 10.0]])],
            [np.zeros((0, 4))],
        ]
        detections = [
            [np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])],
            [np.array([[20.0, 20.0, 30.0, 30.0, 0.8]])],
        ]

        with (
            mock.patch.object(
                detection_eval,
                "_get_annotations",
                return_value=annotations,
            ),
            mock.patch.object(
                detection_eval,
                "_get_detections",
                return_value=detections,
            ),
        ):
            _, precision, recall = detection_eval.evaluateMAP_simple(
                generator,
                dataloader_val=object(),
                sampler_val=object(),
                model=object(),
                iou_threshold=0.5,
            )

        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 1.0)

    def test_optional_diagnostics_count_all_images(self):
        """Count diagnostics include predictions made on empty images."""
        generator = _Generator(num_classes=1, num_images=2)
        annotations = [
            [np.array([[0.0, 0.0, 10.0, 10.0]])],
            [np.zeros((0, 4))],
        ]
        detections = [
            [np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])],
            [np.array([[20.0, 20.0, 30.0, 30.0, 0.8]])],
        ]

        with (
            mock.patch.object(
                detection_eval, "_get_annotations", return_value=annotations
            ),
            mock.patch.object(
                detection_eval, "_get_detections", return_value=detections
            ),
        ):
            _, _, _, diagnostics = detection_eval.evaluateMAP_simple(
                generator,
                dataloader_val=object(),
                sampler_val=object(),
                model=object(),
                iou_threshold=0.5,
                return_diagnostics=True,
            )

        self.assertEqual(diagnostics.ground_truth_objects, 1)
        self.assertEqual(diagnostics.matched_objects, 1)
        self.assertEqual(diagnostics.false_positives, 1)

    def test_duplicate_detection_counts_as_false_positive(self):
        """A second prediction for the same GT object lowers standard precision."""
        generator = _Generator(num_classes=1, num_images=1)
        annotations = [[np.array([[0.0, 0.0, 10.0, 10.0]])]]
        detections = [
            [
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9],
                        [0.0, 0.0, 10.0, 10.0, 0.8],
                    ]
                )
            ]
        ]

        with (
            mock.patch.object(detection_eval, "_get_annotations", return_value=annotations),
            mock.patch.object(detection_eval, "_get_detections", return_value=detections),
        ):
            mean_ap, precision, recall = detection_eval.evaluateMAP_simple(
                generator,
                dataloader_val=object(),
                sampler_val=object(),
                model=object(),
                iou_threshold=0.5,
            )

        self.assertAlmostEqual(mean_ap, 1.0)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 1.0)

    def test_multiclass_precision_and_recall_are_aggregated(self):
        """Returned precision/recall summarize all classes, not only the last one."""
        generator = _Generator(num_classes=2, num_images=2)
        annotations = [
            [
                np.array([[0.0, 0.0, 10.0, 10.0]]),
                np.zeros((0, 4)),
            ],
            [
                np.zeros((0, 4)),
                np.array([[0.0, 0.0, 10.0, 10.0]]),
            ],
        ]
        detections = [
            [
                np.array([[0.0, 0.0, 10.0, 10.0, 0.9]]),
                np.zeros((0, 5)),
            ],
            [
                np.zeros((0, 5)),
                np.array(
                    [
                        [0.0, 0.0, 10.0, 10.0, 0.9],
                        [20.0, 20.0, 30.0, 30.0, 0.8],
                    ]
                ),
            ],
        ]

        with (
            mock.patch.object(detection_eval, "_get_annotations", return_value=annotations),
            mock.patch.object(detection_eval, "_get_detections", return_value=detections),
        ):
            _, precision, recall = detection_eval.evaluateMAP_simple(
                generator,
                dataloader_val=object(),
                sampler_val=object(),
                model=object(),
                iou_threshold=0.5,
            )

        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 1.0)

    def test_pr_curve_handles_ground_truth_without_detections(self):
        """An empty prediction set produces zero metrics without a plot crash."""
        generator = _Generator(num_classes=1, num_images=1)
        annotations = [[np.array([[0.0, 0.0, 10.0, 10.0]])]]
        detections = [[np.zeros((0, 5))]]
        plot = mock.Mock()

        with (
            mock.patch.object(detection_eval, "_get_annotations", return_value=annotations),
            mock.patch.object(detection_eval, "_get_detections", return_value=detections),
            mock.patch.object(detection_eval.plt, "plot", plot),
            mock.patch.object(detection_eval.plt, "savefig"),
        ):
            mean_ap, precision, recall = detection_eval.evaluateMAP_simple(
                generator,
                dataloader_val=object(),
                sampler_val=object(),
                model=object(),
                iou_threshold=0.5,
                generate_PR_curve=True,
            )

        self.assertEqual(mean_ap, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        plotted_recall, plotted_precision = plot.call_args.args
        self.assertEqual(plotted_recall.size, 0)
        self.assertEqual(plotted_precision.size, 0)


if __name__ == "__main__":
    unittest.main()
