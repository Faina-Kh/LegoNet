"""Shared keypoint-map and point-based suppression utilities."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np


class KeypointUtilitiesMixin:
    """Provide behavior shared by the per-object model implementations."""

    @staticmethod
    def image_output_shape(image_shape: Sequence[int], pyramid_level: int = 3) -> np.ndarray:
        """Return the keypoint-map shape for an input image."""
        return (np.array(image_shape[:2]) + 2**pyramid_level - 1) // (2**pyramid_level)

    @staticmethod
    def images_ratios(image_shape: Sequence[int], output_shape: np.ndarray) -> np.ndarray:
        """Return height and width scaling ratios between image and output map."""
        return output_shape / np.array(image_shape[:2])

    @staticmethod
    def create_gausian_mask(
        center_point: Sequence[float],
        nCols: int,
        nRows: int,
        q: float = 99,
        radius: tuple[float, float] = (5, 5),
    ) -> np.ndarray:
        """Create the legacy Gaussian keypoint target mask."""
        s = 3
        x = np.tile(range(nCols), (nRows, 1))
        y = np.tile(np.reshape(range(nRows), (nRows, 1)), (1, nCols))

        x2 = (((x - round(center_point[0])) * s) / radius[0]) ** 2
        y2 = (((y - round(center_point[1])) * s) / radius[1]) ** 2
        probabilities = np.exp(-0.5 * (x2 + y2))
        probabilities[np.where(probabilities < np.percentile(probabilities, q))] = 0
        probabilities = probabilities / np.max(probabilities)
        if not np.isfinite(probabilities).all():
            print("divide by zero")
        return probabilities

    def compute_keypoints_targets_multi_maps(
        self,
        image_shape: Sequence[int],
        annotations_points_centers_a: list[dict[str, float]],
        radius: tuple[float, float] = (5, 5),
        pyramid_level: int = 3,
    ):
        """Create a target map by combining Gaussian masks for all keypoints."""
        annotations_points_centers = copy.deepcopy(annotations_points_centers_a)
        output_shape = self.image_output_shape(image_shape, pyramid_level=pyramid_level)
        image_ratio = self.images_ratios(image_shape, output_shape)

        if len(annotations_points_centers) == 0:
            return [np.zeros(output_shape)]

        annotations = np.zeros(output_shape)
        for point in annotations_points_centers:
            point["y"] *= image_ratio[0]
            point["x"] *= image_ratio[1]
            current_point = [point["x"], point["y"]]
            gaussian_map = self.create_gausian_mask(
                current_point, output_shape[1], output_shape[0], radius=radius
            )
            annotations = np.maximum(annotations, gaussian_map)
            if np.isnan(annotations).any():
                raise RuntimeError("nan was found")

        return annotations

    @staticmethod
    def nmcs(predicted_boxes, relevant_points: list[dict[str, list[float]]]) -> list[bool]:
        """Suppress boxes whose contained keypoints are a subset of another box."""
        non_supressed_indices = [True for _ in range(predicted_boxes.shape[0])]

        for i in range(predicted_boxes.shape[0]):
            current_points = relevant_points[i]
            if len(current_points["x"]) == 0:
                continue

            for j in range(predicted_boxes.shape[0]):
                candidate_points = relevant_points[j]
                if i == j or len(candidate_points["x"]) == 0 or not non_supressed_indices[j]:
                    continue

                common_points_count = 0
                for x1, y1 in zip(candidate_points["x"], candidate_points["y"]):
                    for x2, y2 in zip(current_points["x"], current_points["y"]):
                        if x1 == x2 and y1 == y2:
                            common_points_count += 1
                if common_points_count == len(candidate_points["x"]):
                    non_supressed_indices[j] = False

        return non_supressed_indices
