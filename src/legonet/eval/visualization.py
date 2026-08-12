"""Optional image artifacts for per-object evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

def _number(value: Any) -> float:
    """Convert a scalar tensor or numeric value to ``float``."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def scaled_box(box: Sequence[Any], scale: Any) -> Tuple[float, float, float, float]:
    """Convert a detector box to original-image coordinates."""
    image_scale = _number(scale[0])
    return tuple(_number(coordinate) / image_scale for coordinate in box[:4])


def _matched_gt_box(
    gt_boxes: Any,
    gt_box_id: int,
    scale: Any,
    roots_attributes: bool,
) -> Tuple[float, float, float, float] | None:
    """Return the matched GT box in original-image coordinates."""
    if gt_box_id == -1 or len(gt_boxes) == 0:
        return None
    gt_box = gt_boxes[gt_boxes[:, 5] == gt_box_id][0]
    return scaled_box(gt_box, scale)


def _crop_image(bbox_crop: Any, unnormalize: Callable[[Any], Any]) -> Image.Image:
    """Convert one normalized CHW crop to an RGB image."""
    crop = bbox_crop
    for method_name in ("cpu", "clone", "detach"):
        if hasattr(crop, method_name):
            crop = getattr(crop, method_name)()
    crop_array = np.asarray(255 * unnormalize(crop))
    crop_array = np.clip(crop_array, 0, 255)
    crop_array = np.transpose(crop_array, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(crop_array, mode="RGB")


def _visualize_keypoint_heatmaps(*args: Any) -> None:
    """Load the optional keypoint renderer only when maps are requested."""
    from legonet.eval.KP_detection_eval import visualize_KeyPointsHeatmaps

    visualize_KeyPointsHeatmaps(*args)


def save_detection_overview(
    image_path: str,
    image_name: str,
    predicted_boxes: Sequence[Sequence[Any]],
    gt_boxes: Any,
    point_annotations: Sequence[Mapping[str, Any]],
    scale: Any,
    have_gt: bool,
    draw_path: str,
    gt_path: str,
    line_width: int,
    point_radius: int,
    draw_detection_overview: bool = True,
    draw_gt_only: bool = False,
) -> None:
    """Save original-image overlays for points, GT boxes, and predictions."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if have_gt:
        for point in point_annotations:
            draw.ellipse(
                (
                    point["x"] - point_radius,
                    point["y"] - point_radius,
                    point["x"] + point_radius,
                    point["y"] + point_radius,
                ),
                fill="black",
                width=line_width,
            )

        for index, box in enumerate(gt_boxes):
            coordinates = scaled_box(box, scale)
            draw.rectangle(
                (coordinates[:2], coordinates[2:]),
                outline="blue",
                width=line_width,
            )
            if draw_gt_only:
                gt_image = Image.open(image_path)
                gt_draw = ImageDraw.Draw(gt_image)
                gt_draw.rectangle(
                    (coordinates[:2], coordinates[2:]),
                    outline="blue",
                    width=line_width,
                )
                image_stem = Path(image_name).stem
                gt_image.save(Path(gt_path) / f"{image_stem}_gt_box_{index}.jpg")

        if draw_gt_only:
            image.save(Path(gt_path) / image_name)

    if draw_detection_overview:
        for box in predicted_boxes:
            coordinates = tuple(_number(value) for value in box[:4])
            draw.rectangle(
                (coordinates[:2], coordinates[2:]),
                outline="red",
                width=line_width,
            )
        image.save(Path(draw_path) / image_name)


def save_object_visualizations(
    image_path: str,
    image_name: str,
    crop_index: int,
    bbox_crop: Any,
    predicted_box: Sequence[Any],
    scale: Any,
    gt_boxes: Any,
    gt_box_id: int,
    roots_attributes: bool,
    image_points: Sequence[Mapping[str, Any]],
    crop_points: Sequence[Mapping[str, Any]],
    has_positive_target: bool,
    predicted_boxes_path: str,
    crops_path: str,
    line_width: int,
    unnormalize: Callable[[Any], Any],
) -> Image.Image:
    """Save the source-image overlay and annotated predicted crop."""
    source_image = Image.open(image_path)
    source_draw = ImageDraw.Draw(source_image)
    x1, y1, x2, y2 = scaled_box(predicted_box, scale)
    source_draw.rectangle(
        ((x1, y1), (x2, y2)), outline="red", width=line_width
    )

    if has_positive_target:
        radius = 2
        for point in image_points:
            if x1 <= point["x"] <= x2 and y1 <= point["y"] <= y2:
                source_draw.ellipse(
                    (
                        point["x"] - radius,
                        point["y"] - radius,
                        point["x"] + radius,
                        point["y"] + radius,
                    ),
                    fill="black",
                    width=line_width,
                )

    matched_box = _matched_gt_box(
        gt_boxes, gt_box_id, scale, roots_attributes
    )
    if matched_box is not None:
        source_draw.rectangle(
            (matched_box[:2], matched_box[2:]),
            outline="blue",
            width=line_width,
        )

    image_stem = Path(image_name).stem
    source_image.save(
        Path(predicted_boxes_path) / f"{image_stem}_predicted_BBOX_{crop_index}.jpg"
    )

    crop_image = _crop_image(bbox_crop, unnormalize)
    crop_draw = ImageDraw.Draw(crop_image)
    for point in crop_points:
        x_coordinate = int(point["x"])
        y_coordinate = int(point["y"])
        crop_draw.ellipse(
            (
                (x_coordinate - 5, y_coordinate - 5),
                (x_coordinate + 5, y_coordinate + 5),
            ),
            fill="black",
            width=line_width,
        )
    crop_image.save(Path(crops_path) / f"{image_stem}_crop_{crop_index}.jpg")
    return crop_image.copy()


def save_keypoint_heatmap(
    image_name: str,
    crop_index: int,
    crop_image: Image.Image,
    point_maps: Sequence[Any],
    predicted_maps: Sequence[Any],
    maps_index: int,
    draw_maps: bool,
    maps_path: str,
) -> int:
    """Optionally save the fifth keypoint heatmap and advance its map index."""
    true_maps = [
        np.asarray(point_maps[index][crop_index]).copy()
        for index in (1, 2, 3, 4, 5)
    ]
    output_maps = []
    if len(predicted_maps[crop_index]) > 1:
        output_maps = [predicted_maps[maps_index][index] for index in range(5)]

    if draw_maps:
        heatmap_index = 4
        predicted_map = (
            output_maps[heatmap_index].cpu()
            if output_maps and hasattr(output_maps[heatmap_index], "cpu")
            else output_maps[heatmap_index] if output_maps else None
        )
        image_stem = Path(image_name).stem
        map_name = f"{image_stem}_crop_{crop_index}_map_{heatmap_index + 1}"
        _visualize_keypoint_heatmaps(
            predicted_map,
            true_maps[heatmap_index],
            image_stem,
            map_name,
            crop_image,
            maps_path,
        )
    return maps_index + 1
