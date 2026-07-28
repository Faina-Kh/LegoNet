"""Tests for splitting current full-model checkpoints."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from legonet.checkpoint_conversion import (
    combine_partial_state_dicts,
    convert_full_checkpoint,
    estimator_module_names,
    split_full_state_dict,
)


def _module_state(*module_names):
    return {f"{name}.weight": object() for name in module_names}


class CheckpointConversionTests(unittest.TestCase):
    """Verify generic module naming and architecture-safe extraction."""

    def test_empty_attribute_names_use_single_estimator(self):
        self.assertEqual(estimator_module_names([]), ["estimator"])

    def test_partial_counting_checkpoints_are_combined(self):
        detector = _module_state("backbone_1", "find_1", "where")
        head = _module_state("backbone_2", "estimator")

        combined = combine_partial_state_dicts(
            detector,
            head,
            network_type="per_object_counting",
            estimate_type="reg_fpn_p3_p7_min_sig",
        )

        self.assertEqual(
            set(combined),
            {
                "bbox_detection.backbone_1.weight",
                "bbox_detection.find_1.weight",
                "bbox_detection.where.weight",
                "backbone_2.weight",
                "estimator.weight",
            },
        )

    def test_incompatible_partial_head_is_rejected_when_combining(self):
        detector = _module_state("backbone_1", "find_1", "where")
        keypoint_head = _module_state("backbone_2", "find_2", "estimator")

        with self.assertRaisesRegex(ValueError, "unexpected modules.*find_2"):
            combine_partial_state_dicts(
                detector,
                keypoint_head,
                network_type="per_object_counting",
                estimate_type="reg_fpn_p3_p7_min_sig",
            )

    def test_missing_defined_module_is_rejected_when_combining(self):
        detector = _module_state("backbone_1", "find_1", "where")
        incomplete_head = _module_state(
            "backbone_2",
            "estimator_length",
            "estimator_diameter",
            "estimator_color",
        )

        with self.assertRaisesRegex(ValueError, "missing modules.*find_2"):
            combine_partial_state_dicts(
                detector,
                incomplete_head,
                network_type="per_object_attributes",
                estimate_type="reg_fpn_p3_p7_min_sig",
                attribute_names=["length", "diameter", "color"],
            )

    def test_combined_counting_checkpoint_round_trips(self):
        detector = _module_state("backbone_1", "find_1", "where")
        head = _module_state("backbone_2", "find_2", "estimator")
        combined = combine_partial_state_dicts(
            detector,
            head,
            network_type="per_object_counting",
            estimate_type="withKeyPoints",
        )

        split_detector, split_head = split_full_state_dict(
            combined,
            network_type="per_object_counting",
            estimate_type="withKeyPoints",
        )

        self.assertEqual(split_detector, detector)
        self.assertEqual(split_head, head)

    def test_bbox_detection_is_not_a_conversion_network(self):
        with self.assertRaisesRegex(ValueError, "Unsupported network type"):
            split_full_state_dict(
                _module_state("backbone_1", "find_1", "where"),
                network_type="bbox_detection",
            )

    def test_attribute_names_create_named_estimators(self):
        self.assertEqual(
            estimator_module_names(["length", "diameter", "color"]),
            ["estimator_length", "estimator_diameter", "estimator_color"],
        )

    def test_empty_multibranch_attributes_fail_without_inventing_estimator(self):
        detector = _module_state("backbone_1", "find_1", "where")
        head = _module_state(
            "backbone_2",
            "find_2",
            "estimator_length",
            "estimator_diameter",
            "estimator_color",
            "backbone_2_b",
            "find_2_b",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Attribute names are required.*has no 'estimator' module",
        ):
            combine_partial_state_dicts(
                detector,
                head,
                network_type="per_object_attributes_multibranch",
                estimate_type="withKeyPoints",
                attribute_names=[],
            )

    def test_regression_counting_checkpoint_is_split(self):
        state_dict = {
            **_module_state("backbone_2", "estimator"),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }

        detector, head = split_full_state_dict(
            state_dict,
            network_type="per_object_counting",
            estimate_type="reg_fpn_p3_p7_min_sig",
        )

        self.assertEqual(
            set(detector),
            {"backbone_1.weight", "find_1.weight", "where.weight"},
        )
        self.assertEqual(set(head), {"backbone_2.weight", "estimator.weight"})

    def test_incompatible_keypoint_module_is_rejected(self):
        state_dict = {
            **_module_state("backbone_2", "find_2", "estimator"),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }

        with self.assertRaisesRegex(ValueError, "unexpected modules.*find_2"):
            split_full_state_dict(
                state_dict,
                network_type="per_object_counting",
                estimate_type="reg_fpn_p3_p7_min_sig",
            )

    def test_named_attribute_estimators_are_extracted(self):
        attributes = ["length", "diameter", "color"]
        state_dict = {
            **_module_state(
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }

        _, head = split_full_state_dict(
            state_dict,
            network_type="per_object_attributes",
            estimate_type="withKeyPoints",
            attribute_names=attributes,
        )

        self.assertEqual(
            set(name.split(".")[0] for name in head),
            {
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            },
        )

    def test_defined_find_module_is_retained_for_regression_attributes(self):
        attributes = ["length", "diameter", "color"]
        state_dict = {
            **_module_state(
                "backbone_2",
                "find_2",
                "estimator_length",
                "estimator_diameter",
                "estimator_color",
            ),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }

        _, head = split_full_state_dict(
            state_dict,
            network_type="per_object_attributes",
            estimate_type="reg_fpn_p3_p7_min_sig",
            attribute_names=attributes,
        )

        self.assertIn("find_2", {name.split(".")[0] for name in head})

    def test_detector_output_can_be_omitted(self):
        state_dict = {
            **_module_state("backbone_2", "estimator"),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }
        fake_torch = types.ModuleType("torch")
        fake_torch.load = mock.Mock(return_value=state_dict)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full.pt"
            source.touch()
            head_output = Path(directory) / "head.pt"
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch(
                "legonet.checkpoint_conversion._save_state_dicts_atomically"
            ) as save:
                detector_path, returned_head = convert_full_checkpoint(
                    full_weights_file=source,
                    detector_output_file=None,
                    per_object_output_file=head_output,
                    network_type="per_object_counting",
                    estimate_type="reg_fpn_p3_p7_min_sig",
                )

        self.assertIsNone(detector_path)
        self.assertEqual(returned_head, head_output)
        save.assert_called_once()
        outputs = save.call_args[0][0]
        overwrite = save.call_args[0][1]
        self.assertEqual([path for path, _ in outputs], [head_output])
        self.assertFalse(overwrite)

    def test_empty_path_object_does_not_select_detector_output(self):
        state_dict = {
            **_module_state("backbone_2", "estimator"),
            **{
                "bbox_detection.backbone_1.weight": object(),
                "bbox_detection.find_1.weight": object(),
                "bbox_detection.where.weight": object(),
            },
        }
        fake_torch = types.ModuleType("torch")
        fake_torch.load = mock.Mock(return_value=state_dict)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full.pt"
            source.touch()
            head_output = Path(directory) / "head.pt"
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch(
                "legonet.checkpoint_conversion._save_state_dicts_atomically"
            ) as save:
                detector_path, _ = convert_full_checkpoint(
                    full_weights_file=source,
                    detector_output_file=Path(),
                    per_object_output_file=head_output,
                    network_type="per_object_counting",
                    estimate_type="reg_fpn_p3_p7_min_sig",
                )

        self.assertIsNone(detector_path)
        self.assertEqual(
            [path for path, _ in save.call_args[0][0]],
            [head_output],
        )


if __name__ == "__main__":
    unittest.main()
