"""Tests for per-image attribute evaluation helpers."""

import unittest

import numpy as np
from legonet.eval.scalars import first_scalar


class FirstScalarTests(unittest.TestCase):
    """Verify scalar extraction across annotation layouts."""

    def test_extracts_scalar_from_multi_attribute_array(self) -> None:
        annotation = np.array([[[7.0, 8.0, 9.0, 10.0]]])
        self.assertEqual(first_scalar(annotation), 7.0)

    def test_extracts_scalar_from_numpy_array(self) -> None:
        annotation = np.array([[3.0]])
        self.assertEqual(first_scalar(annotation), 3.0)


if __name__ == "__main__":
    unittest.main()
