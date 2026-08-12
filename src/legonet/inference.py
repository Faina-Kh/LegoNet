"""Inference and visualization orchestration for LegoNet models."""

import os
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from legonet import config
from legonet import utils
from legonet.eval import detection_eval, per_image_attribute_eval
from legonet.eval import perObject_eval
from legonet.my_dataloader import UnNormalizer
from legonet.utils import printf


def _group_cache_key(inputs: Any) -> Hashable:
    """Return a stable validation-sample key from a combined-model input."""
    group_index = inputs[2]
    if hasattr(group_index, "detach"):
        group_index = group_index.detach()
    if hasattr(group_index, "cpu"):
        group_index = group_index.cpu()
    if hasattr(group_index, "tolist"):
        group_index = group_index.tolist()
    if isinstance(group_index, list):
        return tuple(group_index)
    if isinstance(group_index, tuple):
        return group_index
    return group_index


class _CachedCombinedModel:
    """Cache one combined-model output per validation sample."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._outputs: dict[Hashable, Any] = {}

    def __call__(self, inputs: Any) -> Any:
        key = _group_cache_key(inputs)
        if key not in self._outputs:
            self._outputs[key] = self._model(inputs)
        return self._outputs[key]

    def eval(self) -> "_CachedCombinedModel":
        """Put the wrapped model in evaluation mode."""
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "_CachedCombinedModel":
        """Restore the wrapped model's training mode."""
        self._model.train(mode)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate model attributes not owned by the cache."""
        return getattr(self._model, name)


def visualize_bboxes(
    dataloader_val: Any,
    sampler_val: Any,
    dataset_val: Any,
    model: Any,
    unnormalize: Any,
    have_ground_truth: bool = True,
) -> None:
    """Save standalone-detector prediction overlays, with GT when available."""
    output_directory = os.path.join(
        config.DrawProperties.save_img_path,
        "BBOX visualization",
    )
    os.makedirs(output_directory, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 14)
    print()
    for index, data in enumerate(dataloader_val):
        group_index = sampler_val.groups[index]
        image_id = dataset_val.image_ids[group_index[0]]
        image_name = dataset_val.img_info[image_id]["name"]

        with torch.no_grad():
            if config.General.NETWORK_TYPE == config.NetworkType.detection:
                detection_outputs = model(
                    [
                        data["img"].to(config.General.device).float(),
                        [data["bbox_annot"], None],
                        None,
                    ]
                )
            elif (
                config.General.NETWORK_TYPE
                == config.NetworkType.detection_and_estimation
            ):
                if "points_annot" in data:
                    annotations = [data["bbox_annot"], data["points_annot"]]
                else:
                    annotations = [data["bbox_annot"], None]

                detection_outputs, _, _, _, _ = model(
                    [
                        data["img"].to(config.General.device).float(),
                        annotations,
                        torch.tensor(group_index),
                    ]
                )
            else:
                continue

            scores, classification, transformed_anchors = detection_outputs
            selected_indices = np.where(scores.cpu() > config.Detection.min_score)
            image_array = np.array(255 * unnormalize(data["img"][0, :, :, :])).copy()
            image_array[image_array < 0] = 0
            image_array[image_array > 255] = 255
            image_array = np.transpose(image_array, (1, 2, 0))

            image = Image.fromarray(np.uint8(image_array))
            draw = ImageDraw.Draw(image)

            for prediction_index in range(selected_indices[0].shape[0]):
                annotation = transformed_anchors[
                    selected_indices[0][prediction_index], :
                ]
                x1, y1, x2, y2 = [int(value) for value in annotation]
                draw.rectangle(
                    ((x1, y1), (x2, y2)),
                    outline="red",
                    width=config.DrawProperties.LINE_WIDTH,
                )

            if have_ground_truth:
                for annotation in data["bbox_annot"].numpy()[0]:
                    if annotation[0] != -1:
                        x1, y1, x2, y2 = [int(value) for value in annotation[:4]]
                        draw.rectangle(
                            ((x1, y1), (x2, y2)),
                            outline="blue",
                            width=config.DrawProperties.LINE_WIDTH,
                        )

            legend = (
                f"{image_name} | predictions: red | ground truth: blue"
                if have_ground_truth
                else f"{image_name} | predictions: red"
            )
            text_box = font.getbbox(legend)
            label_width = max(200, text_box[2] - text_box[0] + 10)
            label_height = max(20, text_box[3] - text_box[1] + 8)
            label_image = Image.new("RGBA", (label_width, label_height), "black")
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((5, 2), legend, fill="white", font=font)
            image.paste(label_image, (0, 0))
            suffix = (
                "_predictions_and_gt.png"
                if have_ground_truth
                else "_predictions.png"
            )
            output_name = f"{Path(image_name).stem}{suffix}"
            image.save(os.path.join(output_directory, output_name))


def _evaluate_detection(
    args: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
) -> Any:
    """Run the configured detection inference evaluation."""
    detection_metrics = None
    if args.evaluate_detection and args.have_GT:
        print()
        print("------------------Object detection evaluation:------------------\n")
        if args.eval_detection_params:
            all_average_precisions = detection_eval.evaluate_detection_params(
                dataset_val,
                dataloader_val,
                sampler_val,
                model,
                iou_threshold=config.Detection.iou_threshold_list,
                score_threshold=config.Detection.min_score_list,
                save_path=args.test_dir,
                show_PR_curve=True,
            )
            print("iou, score, class_mAP")
            for average_precision in all_average_precisions:
                print(average_precision)
        else:
            print(
                f"Results for min score: {config.Detection.min_score}, "
                f"iou_threshold: {config.Detection.iou_threshold}"
            )
            detection_metrics = detection_eval.evaluateMAP_simple(
                dataset_val,
                dataloader_val,
                sampler_val,
                model,
                score_threshold=config.Detection.min_score,
                iou_threshold=config.Detection.iou_threshold,
                generate_PR_curve=True,
                return_diagnostics=True,
            )
            mean_average_precision, precision, recall = detection_metrics[:3]
            result_line = (
                f"mAP = {mean_average_precision:.3f}, "
                f"precision = {precision:.3f}, recall = {recall:.3f}"
            )
            print()
            print(result_line)

    if (
        config.General.to_draw
        and config.General.NETWORK_TYPE == config.NetworkType.detection
    ):
        visualize_bboxes(
            dataloader_val,
            sampler_val,
            dataset_val,
            model,
            unnormalize=UnNormalizer(),
            have_ground_truth=args.have_GT,
        )
    return detection_metrics


def _evaluate_attributes(
    args: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
    detection_metrics: Any = None,
) -> None:
    """Evaluate per-object attributes produced by a combined model."""
    if not args.evaluate_per_object:
        return

    print("------------------Attribute estimation evaluation:------------------\n")
    output = perObject_eval.eval(
        dataset_val,
        dataloader_val,
        sampler_val,
        model,
        to_draw=config.General.to_draw,
        verbose=True,
        draw_path=config.DrawProperties.save_img_path,
        print_to_files=True,
        args=args,
        detection_metrics=detection_metrics,
        evaluate_points=(
            getattr(args, "have_GT", False)
            and config.AttributeEstimation.estimate_type == "withKeyPoints"
        ),
    )
    #relative_error = output[0] if len(output) > 0 else -1
    #utils.printf("rel error: %.3f \n", relative_error)


def run_inference(
    args: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
) -> None:
    """Run inference for the configured network type."""
    model.training = False
    model.eval()

    if config.General.NETWORK_TYPE in (
        config.NetworkType.detection,
        config.NetworkType.detection_and_estimation,
    ):
        evaluation_model = model
        if (
            config.General.NETWORK_TYPE
            == config.NetworkType.detection_and_estimation
            and args.evaluate_detection
            and args.have_GT
            and args.evaluate_per_object
        ):
            evaluation_model = _CachedCombinedModel(model)

        detection_metrics = None
        if (
            config.General.NETWORK_TYPE == config.NetworkType.detection
            or args.evaluate_detection
        ):
            detection_metrics = _evaluate_detection(
                args,
                dataset_val,
                dataloader_val,
                sampler_val,
                evaluation_model,
            )
            print()
        _evaluate_attributes(
            args,
            dataset_val,
            dataloader_val,
            sampler_val,
            evaluation_model,
            detection_metrics=detection_metrics,
        )
        return

    utils.printf("------------------Attribute estimation evaluation:------------------\n")
    if config.General.NETWORK_TYPE in (
        config.NetworkType.per_image_estimation_regression,
        config.NetworkType.per_image_estimation_keypoints,
    ):
        relative_error = per_image_attribute_eval.evaluate(
            dataloader_val,
            dataset_val,
            model,
            args,
        )
