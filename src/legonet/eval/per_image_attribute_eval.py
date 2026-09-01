"""Evaluation and checkpoint metrics for per-image attribute models."""

from __future__ import annotations

import torch
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from legonet import config
import csv
from PIL import Image
from thop import profile, clever_format

from legonet.eval.KP_detection_eval import (
    calc_points_recall_precision_ap,
    points_detection_t_p,
    process_keypoint_map_for_evaluation,
    visualize_KeyPointsHeatmaps,
)
from legonet.eval.detection_eval import plot_PR_curve
from legonet.eval.reporting import format_keypoint_summary
from legonet.eval.scalars import first_scalar
from legonet.eval.numeric_metrics import (
    sum_of_absolute_differences,
    sum_of_differences,
)
from legonet.eval.regression_metrics import compute_regression_metrics

@dataclass(frozen=True)
class PerImageCheckpointMetrics:
    """Named error used for per-image checkpoint selection."""

    metric_name: str
    metric_value: Optional[float]


def evaluate(
    dataloader: Any,
    dataset: Any,
    model: Any,
    args: Any,
    do_profile: bool = False,
) -> Optional[float]:
    """Evaluate the configured per-image attribute model."""

    print("\nStart evaluation")
    model.eval()

    with torch.no_grad():

        all_GT_values = []
        all_predicted_values = []
        T, P = [], []
        protocol_comparison = {}
        all_rel_error = []
        predicted_maps = None

        print("Per-image values:")
        if args.have_GT and model.estimator.binary_model:
            print("Image | GT | Predicted")
        elif args.have_GT:
            print("Image | GT | Predicted | Absolute difference | Relative error (MRD)")
        else:
            print("Image | Predicted")

        for iter_num, data in enumerate(dataloader):

            full_rgbImage_name = dataset.bgr_images_names[dataloader.batch_sampler.groups[iter_num][0]]

            Image_name = Path(full_rgbImage_name).stem

            if args.estimate_type == "reg_fpn_p3_p7_min_sig":
                if args.have_GT:
                    GT = first_scalar(data['annot'][0])
                    prediction = float(model([data['img'].to(config.General.device).float(), data['annot']])[0].squeeze().item())
                else:
                    prediction = float(model([data['img'].to(config.General.device).float()])[0].squeeze().item())

                if prediction<0:
                    prediction=0

            else:
                if args.have_GT:
                    GT = first_scalar(data['annot'][0])
                    count_outputs = model([data['img'].to(config.General.device).float(), data['annot']])

                    ###########################################################################################################
                    if iter_num == 0 and do_profile:
                        print("Get FLOPS for per_image_estimation with keypoints:")
                        # Use thop to profile the model
                        input = [data['img'].to(config.General.device).float(), data['annot']]
                        flops, params = profile(model, inputs=(input,))

                        # Print the estimated FLOPS and parameters
                        flops_str, params_str = clever_format([flops, params], "%.3f")
                        print(f"FLOPS: {flops_str}")
                        print(f"Params: {params_str}")

                    ###########################################################################################################

                else:
                    count_outputs = model([data['img'].to(config.General.device).float()])

                prediction = count_outputs[0].squeeze().item() #float(count_outputs[0].cpu().detach().numpy())
                if model.estimator.binary_model:
                    prediction=np.round(prediction)


                # get only the prediction maps
                predicted_maps = count_outputs[1:5]
                predicted_maps.append(count_outputs[6])

                #---------------------------------------
                # import matplotlib.pyplot as plt
                #
                # i=0
                # my_tensor = predicted_maps[i]
                # tensor_2d = my_tensor.squeeze().detach().cpu().numpy()
                #
                # plt.imsave(config.DrawProperties.maps_path + '/' + 'pred_map_'+str(i)+'.png' , tensor_2d)

                # ---------------------------------------

            if args.estimate_type == "withKeyPoints" and config.General.to_draw:
                img = Image.open(os.path.join(dataset.base_dir, full_rgbImage_name))
                if args.have_GT and args.val_csv_leaf_location_file != "":
                    gt_maps = data['annot'][1:6]

                else:
                    GT = None
                    gt_maps = None

                for i in [4]:  # range(5): # choose which heatmaps to visualize, numbered from 0 to 4
                    map_name = Image_name + '_map_' + str(i + 1)
                    predicted_map = predicted_maps[i].cpu().numpy()[0]
                    if gt_maps is not None:
                        gt_map = gt_maps[i].cpu().numpy()[0]
                        visualize_KeyPointsHeatmaps(predicted_map, gt_map, Image_name, map_name, img,
                                                    config.DrawProperties.maps_path, prediction, GT,
                                                    heatmap_vmax=1.0)
                    else:
                        visualize_KeyPointsHeatmaps(predicted_map, None, Image_name, map_name, img,
                                                    config.DrawProperties.maps_path, prediction, GT,
                                                    heatmap_vmax=1.0)

            if args.have_GT:
                all_GT_values.append(GT)

            if predicted_maps is not None and args.have_GT:
                if config.AttributeEstimation.calc_det_performance and dataset.csv_leaf_location_file != "":
                    true_map = data['annot'][5]
                    evaluation_map = process_keypoint_map_for_evaluation(
                        model, predicted_maps[-1]
                    )
                    for b in range(evaluation_map.shape[0]):
                        t, p = points_detection_t_p(
                            predicted_maps[-1][b],
                            true_map[b],
                            candidate_threshold=0.0,
                        )
                        T = T + t
                        P = P + p
                        if getattr(args, "compare_keypoint_protocols", False):
                            for protocol_name in (
                                "processed_threshold_0_02",
                                "processed_local_maxima",
                                "raw_nonzero",
                            ):
                                protocol_comparison.setdefault(
                                    protocol_name, {"T": [], "P": []}
                                )
                            protocol_inputs = {
                                "processed_threshold_0_02": points_detection_t_p(
                                    evaluation_map[b], true_map[b]
                                ),
                                "processed_local_maxima": points_detection_t_p(
                                    evaluation_map[b],
                                    true_map[b],
                                    candidate_threshold=None,
                                    local_maxima_only=True,
                                ),
                                "raw_nonzero": points_detection_t_p(
                                    predicted_maps[-1][b],
                                    true_map[b],
                                    candidate_threshold=0.0,
                                ),
                            }
                            for protocol_name, (protocol_t, protocol_p) in protocol_inputs.items():
                                protocol_comparison[protocol_name]["T"].extend(protocol_t)
                                protocol_comparison[protocol_name]["P"].extend(protocol_p)

            if args.have_GT:
                if not model.estimator.binary_model:
                    if GT>0:
                        rel_error = abs(GT - prediction) / GT
                        all_rel_error.append(rel_error)
                    else:
                        rel_error = -1

            all_predicted_values.append(prediction)

            if  model.estimator.binary_model:
                if args.have_GT:

                    print(
                        '{} | {:.3f} | {:.3f}'.format(
                            Image_name, GT, prediction,
                            ))

                else:
                    print(
                        '{} | {:.3f}'.format(
                            Image_name, prediction))

            else:
                if args.have_GT:
                    print(
                        '{} | {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                            Image_name, GT, prediction,
                            abs(GT - prediction), rel_error))

                else:
                    print(
                        '{} | {:.3f}'.format(
                            Image_name, prediction))

        # else:
        #     for index in range(len(dataset)):
        #         data = dataset[index]
        #         count_GT = data['annot'][0][0]
        #
        #         # run network
        #         image = torch.tensor(data['img']).permute(2, 0, 1)
        #         count_outputs = model(image.to(config.General.device).float().unsqueeze(dim=0))
        #         if len(count_outputs):
        #             count_pred = count_outputs[0]
        #             count_pred = count_pred.cpu().item()
        #
        #             full_rgbImage_name = dataset.bgr_images_names[index]
        #             Image_name = full_rgbImage_name.split("_rgb")[0]
        #
        #             all_GT_values.append(count_GT)
        #             all_predicted_values.append(np.round(count_pred))
        #             print(
        #                 'image: {} | GT: {} | predicted: {} ({}) | abs_diff: {}'.format(
        #                     Image_name, int(count_GT), np.round(count_pred), count_pred,
        #                     abs(count_GT - count_pred)))

        if args.have_GT:
            num_of_images = len(all_GT_values)
            valueDiff = sum_of_differences(all_GT_values, all_predicted_values) / num_of_images
            Abs_value_Diff = sum_of_absolute_differences(all_GT_values, all_predicted_values) / num_of_images
            one_minus_fvu = compute_regression_metrics(
                all_GT_values,
                all_predicted_values,
            ).one_minus_fvu
        else:
            print(f"Predictions generated: {len(all_predicted_values)}")

        #if args.dataset_name == "roots":
        if args.have_GT:
            if  model.estimator.binary_model:
                args.per_image_evaluation_summary = (
                    "Evaluation Summary - per-image estimation\n"
                    'Abs_value_Diff: {:.3f} | accuracy: {:.3f} | 1-FVU: {:.3f}'.format(
                        Abs_value_Diff,
                        1-Abs_value_Diff,
                        one_minus_fvu,
                    )
                )

            else:
                mean_rel_error = np.mean(all_rel_error)
                SE = 0
                count_non_zero = 0
                for i in range(len(all_GT_values)):
                    if all_GT_values[i] > 0:
                        count_non_zero += 1
                        SE += (all_GT_values[i] - all_predicted_values[i]) ** 2
                if count_non_zero > 0:
                    MSE = SE / count_non_zero
                else:
                    MSE = None

                args.per_image_evaluation_summary = (
                    "Evaluation Summary - per-image estimation\n"
                    'Abs_value_Diff: {:.3f} | MSE: {:.3f} | MRD (gt > 0): {:.3f} | 1-FVU: {:.3f}'.format(
                        Abs_value_Diff,
                        MSE,
                        mean_rel_error,
                        one_minus_fvu,
                    )
                )
            if config.AttributeEstimation.calc_det_performance and config.General.experiment_path != "" and args.have_GT and config.AttributeEstimation.estimate_type == 'withKeyPoints':
                recall, precision, ap = calc_points_recall_precision_ap(T, P)
                keypoint_summary = format_keypoint_summary(ap).rstrip()
                if getattr(args, "run_script", None) == "Inference":
                    args.per_image_evaluation_summary = "\n".join(
                        part
                        for part in (
                            args.per_image_evaluation_summary,
                            keypoint_summary,
                        )
                        if part
                    )
                else:
                    print(keypoint_summary, end="\n")
                plot_PR_curve(recall, precision, ap,
                              save_path=config.General.files_path, plots_name = 'Points_PR_curve.png')
                if protocol_comparison:
                    comparison_rows = []
                    for protocol_name, values in protocol_comparison.items():
                        protocol_recall, protocol_precision, protocol_ap = (
                            calc_points_recall_precision_ap(
                                values["T"], values["P"]
                            )
                        )
                        comparison_rows.append(
                            (
                                protocol_name,
                                protocol_ap,
                                protocol_recall[-1],
                                protocol_precision[-1],
                                sum(values["T"]),
                                len(protocol_precision),
                            )
                        )
                        plot_PR_curve(
                            protocol_recall,
                            protocol_precision,
                            protocol_ap,
                            save_path=config.General.files_path,
                            plots_name=(
                                f"Points_PR_curve_{protocol_name}.png"
                            ),
                        )
                    comparison_path = os.path.join(
                        config.General.files_path,
                        "keypoint_protocol_comparison.csv",
                    )
                    with open(comparison_path, "w", newline="") as comparison_file:
                        writer = csv.writer(comparison_file)
                        writer.writerow(
                            (
                                "protocol",
                                "mAP",
                                "max_recall",
                                "final_precision",
                                "ground_truth_points",
                                "candidates",
                            )
                        )
                        writer.writerows(comparison_rows)

                # print recall and precision  to csv
                csv_columns = ['recall', 'precision']
                csv_file = os.path.join(config.General.files_path, "parts_recall_precision.csv")
                f = open(csv_file, 'w', newline='')
                with f:
                    writer = csv.writer(f)
                    writer.writerow(csv_columns)
                    for w in range(len(recall)):
                        myrow = []
                        myrow.append(recall[w])
                        myrow.append(precision[w])
                        writer.writerow(myrow)

        # else:
        #     MSE = np.mean((np.array(all_GT_values) - np.array(all_predicted_values)) ** 2)
        #     countAgr = 0
        #     for i in range(num_of_images):
        #         if all_GT_values[i] == all_predicted_values[i]:
        #             countAgr += 1
        #     CountAgreement = countAgr / num_of_images
        #     print('valueDiff: {} | Abs_value_Diff: {} | CountAgreement: {} | MSE {} \n'.format(
        #         valueDiff, Abs_value_Diff, CountAgreement, MSE))

        model.train()

        #if args.dataset_name == "roots":
        if  model.estimator.binary_model:
            if args.have_GT:
                return Abs_value_Diff
            else:
                return None
        else:
            if args.have_GT:
                return mean_rel_error
            else:
                return None
        # else:
        #     return CountAgreement


def evaluate_checkpoint_metrics(
    dataloader: Any,
    dataset: Any,
    model: Any,
    args: Any,
) -> PerImageCheckpointMetrics:
    """Evaluate and name the error used for per-image checkpoint selection."""
    value = evaluate(dataloader, dataset, model, args)
    metric_value = None if value is None else float(value)
    if getattr(model.estimator, "binary_model", False):
        return PerImageCheckpointMetrics(
            metric_name="classification_error_rate",
            metric_value=metric_value,
        )
    if args.network_type == "per_image_estimation":
        return PerImageCheckpointMetrics(
            metric_name="relative_error",
            metric_value=metric_value,
        )
    raise ValueError(
        f"Unsupported per-image attribute network: {args.network_type}"
    )
