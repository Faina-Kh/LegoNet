"""Inference and visualization orchestration for LegoNet models."""

import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from legonet import config
from legonet import utils
from legonet.eval import attribute_estimation_eval, detection_eval
from legonet.eval import perObject_eval
from legonet.my_dataloader import UnNormalizer


def visualize_bboxes(
    dataloader_val: Any,
    sampler_val: Any,
    dataset_val: Any,
    model: Any,
    unnormalize: Any,
) -> None:
    """Save validation images with predicted and ground-truth bounding boxes."""
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

            for annotation in data["bbox_annot"].numpy()[0]:
                if annotation[0] != -1:
                    x1, y1, x2, y2 = [int(value) for value in annotation[:4]]
                    draw.rectangle(
                        ((x1, y1), (x2, y2)),
                        outline="blue",
                        width=config.DrawProperties.LINE_WIDTH,
                    )

            label_image = Image.new("RGBA", (200, 20), "black")
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((0, 0), image_name, font=font)
            image.paste(label_image, (0, 0))
            output_name = image_name.split(".jpg")[0] + "_annot.jpg"
            image.save(os.path.join(config.DrawProperties.save_img_path, output_name))


def _evaluate_detection(
    args: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
) -> None:
    """Run the configured detection inference evaluation."""
    if not (args.evaluate_detection and args.have_GT):
        return

    print()
    print("Object detection evaluation:\n")
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
        return

    print(
        f"Results for min score: {config.Detection.min_score}, "
        f"iou_threshold: {config.Detection.iou_threshold}"
    )
    mean_average_precision, precision, recall = detection_eval.evaluateMAP_simple(
        dataset_val,
        dataloader_val,
        sampler_val,
        model,
        score_threshold=config.Detection.min_score,
        iou_threshold=config.Detection.iou_threshold,
        generate_PR_curve=True,
    )
    result_line = (
        f"mAP = {mean_average_precision:.3f}, precision = {precision:.3f}, "
        f"recall = {recall:.3f}"
    )
    print(result_line)
    with open(args.txt_results, "a") as results_file:
        results_file.write(result_line + "\n")

    if config.General.to_draw:
        visualize_bboxes(
            dataloader_val,
            sampler_val,
            dataset_val,
            model,
            unnormalize=UnNormalizer(),
        )


def _evaluate_combined(
    args: Any,
    dataset_val: Any,
    dataloader_val: Any,
    sampler_val: Any,
    model: Any,
) -> None:
    """Run combined detection and attribute-estimation inference."""
    if not args.evaluate_per_object:
        return

    print("Attribute estimation evaluation:\n")
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
    )
    relative_error = output[0] if len(output) > 0 else -1
    utils.printf("rel error: %.3f \n", relative_error)


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
        if (
            config.General.NETWORK_TYPE == config.NetworkType.detection
            or args.evaluate_detection
        ):
            _evaluate_detection(
                args,
                dataset_val,
                dataloader_val,
                sampler_val,
                model,
            )
            print()
        _evaluate_combined(
            args,
            dataset_val,
            dataloader_val,
            sampler_val,
            model,
        )
        return

    utils.printf("Attribute estimation evaluation:\n")
    if config.General.NETWORK_TYPE in (
        config.NetworkType.per_image_estimation_regression,
        config.NetworkType.per_image_estimation_keypoints,
    ):
        relative_error = attribute_estimation_eval.eval(
            dataloader_val,
            dataset_val,
            model,
            args,
        )
        print("Final avg rel error:", relative_error)
    print("done")
