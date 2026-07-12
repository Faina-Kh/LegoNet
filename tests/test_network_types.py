"""Tests for canonical and legacy model-variant names."""

import unittest
import warnings

from legonet.network_types import canonicalize_network_type


class NetworkTypeTests(unittest.TestCase):
    """Verify compatibility aliases at the public input boundary."""

    def test_canonical_name_is_unchanged(self) -> None:
        name = "per_image_estimation_keypoints"
        self.assertEqual(canonicalize_network_type(name), name)

    def test_legacy_keypoint_name_warns_and_is_canonicalized(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = canonicalize_network_type("counting_lean")

        self.assertEqual(result, "per_image_estimation_keypoints")
        self.assertEqual(caught[0].category, FutureWarning)

    def test_legacy_regression_name_warns_and_is_canonicalized(self) -> None:
        with self.assertWarns(FutureWarning):
            result = canonicalize_network_type("counting_reg")

        self.assertEqual(result, "per_image_estimation_regression")


if __name__ == "__main__":
    unittest.main()
