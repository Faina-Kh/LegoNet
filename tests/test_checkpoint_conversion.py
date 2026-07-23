"""Tests for splitting current full-model checkpoints."""

import unittest

from legonet.checkpoint_conversion import (
    estimator_module_names,
    split_full_state_dict,
)


def _module_state(*module_names):
    return {f"{name}.weight": object() for name in module_names}


class CheckpointConversionTests(unittest.TestCase):
    """Verify generic module naming and architecture-safe extraction."""

    def test_empty_attribute_names_use_single_estimator(self):
        self.assertEqual(estimator_module_names([]), ["estimator"])

    def test_attribute_names_create_named_estimators(self):
        self.assertEqual(
            estimator_module_names(["length", "diameter", "color"]),
            ["estimator_length", "estimator_diameter", "estimator_color"],
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

    def test_unused_find_module_is_omitted_from_regression_attributes(self):
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

        self.assertNotIn("find_2", {name.split(".")[0] for name in head})


if __name__ == "__main__":
    unittest.main()
