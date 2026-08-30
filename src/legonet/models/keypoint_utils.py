"""Shared keypoint-map and point-based suppression utilities."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np


def validate_training_crop_alignment(
    num_of_crops: int,
    **named_batches: Any,
) -> None:
    """Require every prepared training target to match the crop batch size."""
    mismatches = []
    for name, batch in named_batches.items():
        shape = getattr(batch, "shape", None)
        actual_size = int(shape[0]) if shape is not None else len(batch)
        if actual_size != num_of_crops:
            mismatches.append(f"{name}={actual_size}")
    if mismatches:
        details = ", ".join(mismatches)
        raise RuntimeError(
            "Per-object training crop alignment failed: "
            f"expected {num_of_crops} entries, found {details}."
        )


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

    @staticmethod
    def find_points_in_bbox(
        img,
        point_anns: list[dict[str, float]],
        bbox_pred,
        scale: float,
        network_type: str,
    ) -> list[dict[str, list[float]]]:
        """Assign scaled points to predicted boxes using the legacy model rules.

        Attribute models additionally require the point's ``bbox_id`` to match
        the predicted box's GT identifier. Counting models use geometric
        containment only. Point coordinates are scaled in place to preserve the
        behavior of the original model implementations.

        ``img`` remains in the signature for compatibility with existing model
        call sites; the previous implementations only used it for disabled
        visualization code.
        """
        del img
        for point in point_anns:
            point["x"] *= scale
            point["y"] *= scale

        points_by_box = []
        require_matching_bbox_id = network_type in {
            "per_object_attributes",
            "per_object_attributes_multibranch",
        }
        for box in bbox_pred:
            box_x1, box_y1, box_x2, box_y2, box_id = box[:5]
            point_x = []
            point_y = []
            for point in point_anns:
                if require_matching_bbox_id and point["bbox_id"] != box_id:
                    continue
                if (
                    box_x1 <= point["x"] <= box_x2
                    and box_y1 <= point["y"] <= box_y2
                ):
                    point_x.append(point["x"])
                    point_y.append(point["y"])
            points_by_box.append({"x": point_x, "y": point_y})

        return points_by_box
