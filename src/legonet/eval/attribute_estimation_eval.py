"""Compatibility implementation for legacy per-image attribute evaluation.

New training and inference callers use :mod:`per_image_attribute_eval`, which
provides named checkpoint semantics while this implementation is decomposed.
"""

from __future__ import annotations

import torch
import os
import numpy as np
from typing import Any, Optional
from legonet import config
import csv
from PIL import Image
from thop import profile, clever_format

from legonet.eval.KP_detection_eval import points_detection_t_p, calc_points_recall_precision_ap, visualize_KeyPointsHeatmaps
from legonet.eval.detection_eval import plot_PR_curve
from legonet.eval.scalars import first_scalar
from legonet.eval.numeric_metrics import (
    sum_of_absolute_differences,
    sum_of_differences,
)

def eval(
    dataloader: Any,
    dataset: Any,
    model: Any,
    args: Any,
    do_profile: bool = False,
) -> Optional[float]:
    """Evaluate the configured legacy per-image estimation model."""

    print("Start evaluation")
    model.eval()

    # is_per_image_attribute_model = config.Detect_and_Estimate.type in {
    #     "per_image_estimation_keypoints",
    #     "per_image_estimation_regression"
    # }

    with torch.no_grad():

        all_GT_values = []
        all_predicted_values = []
        T, P = [], []
        all_rel_error = []
        predicted_maps = None

        #if is_per_image_attribute_model:
        if args.txt_results != "":
            with open(args.txt_results, 'a') as f:
                if args.have_GT:
                    f.write('image| GT | predicted | abs diff | rel_error')
                else:
                    f.write('image| predicted')
                f.write('\n')

        for iter_num, data in enumerate(dataloader):

            full_rgbImage_name = dataset.bgr_images_names[dataloader.batch_sampler.groups[iter_num][0]]

            if full_rgbImage_name.lower().endswith((".jpg", ".jpeg")):
                Image_name = full_rgbImage_name.split(".jpg")[0]
            elif full_rgbImage_name.lower().endswith(".png"):
                Image_name = full_rgbImage_name.split(".png")[0]

            if args.network_type == "per_image_estimation_regression":
                if args.have_GT:
                    GT = first_scalar(data['annot'][0])
                    prediction = float(model([data['img'].to(config.General.device).float(), data['annot']])[0].squeeze().item())
                else:
                    prediction = float(model([data['img'].to(config.General.device).float()])[0].squeeze().item())

                if prediction<0:
                    prediction=0

            elif args.network_type == "per_image_estimation_keypoints":
                if args.have_GT:
                    GT = first_scalar(data['annot'][0])
                    count_outputs = model([data['img'].to(config.General.device).float(), data['annot']])

                    ###########################################################################################################
                    if iter_num == 0 and do_profile:
                        print("Get FLOPS for per_image_estimation_keypoints:")
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

            if args.network_type == 'per_image_estimation_keypoints' and config.General.to_draw:
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
                                                    config.DrawProperties.maps_path, prediction, GT)
                    else:
                        visualize_KeyPointsHeatmaps(predicted_map, None, Image_name, map_name, img,
                                                    config.DrawProperties.maps_path, prediction, GT)

            if args.have_GT:
                all_GT_values.append(GT)

            if predicted_maps is not None and args.have_GT:
                if config.AttributeEstimation.calc_det_performance and dataset.csv_leaf_location_file != "":
                    if torch.sum(data['annot'][1]).item()> 0: # check that there are gt points
                        true_maps = data['annot'][1:]
                        for m in range(len(predicted_maps)):
                            for b in range(predicted_maps[0].shape[0]):
                                t, p = points_detection_t_p(predicted_maps[m][b], true_maps[m][b])
                                T = T + t
                                P = P + p

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
                        'image: {} | GT: {:.3f} | predicted: {:.3f}'.format(
                            Image_name, GT, prediction,
                            ))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('{} | {:.3f} | {:.3f} '.format(
                                Image_name, GT, prediction
                                ))
                            f.write('\n')

                else:
                    print(
                        'image: {} | predicted: {:.3f}'.format(
                            Image_name, prediction))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('{} | {:.3f} '.format(
                                Image_name, prediction
                            ))
                            f.write('\n')

            else:
                if args.have_GT:
                    print(
                        'image: {} | GT: {:.3f} | predicted: {:.3f} | abs diff: {:.3f} | rel_error: {:.3f}'.format(
                            Image_name, GT, prediction,
                            abs(GT - prediction), rel_error))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('{} | {:.3f} | {:.3f} | {:.3f} |{:.3f}'.format(
                                    Image_name, GT, prediction,
                                    abs(GT - prediction), rel_error))
                            f.write('\n')
                else:
                    print(
                        'image: {} | predicted: {:.3f} '.format(
                            Image_name, prediction))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('{} | {:.3f} |'.format(
                                Image_name, prediction ))
                            f.write('\n')

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
        #                 'image: {} | GT: {} | predicted: {} ({}) | abs diff: {}'.format(
        #                     Image_name, int(count_GT), np.round(count_pred), count_pred,
        #                     abs(count_GT - count_pred)))

        print('\n Summary:')

        if args.have_GT:
            num_of_images = len(all_GT_values)
            valueDiff = sum_of_differences(all_GT_values, all_predicted_values) / num_of_images
            AbsvalueDiff = sum_of_absolute_differences(all_GT_values, all_predicted_values) / num_of_images

        #if args.dataset_name == "roots":
        if args.have_GT:
            if  model.estimator.binary_model:
                print('AbsvalueDiff: {:.3f} | accuracy {:.3f} \n'.format(
                    AbsvalueDiff, 1-AbsvalueDiff))

                if args.txt_results != "":
                    with open(args.txt_results, 'a') as f:
                        f.write('\n')
                        f.write('AbsvalueDiff: {:.3f} | accuracy {:.3f} \n'.format(
                            AbsvalueDiff, 1-AbsvalueDiff ))
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

                print('AbsvalueDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(AbsvalueDiff, MSE,
                                                                                              mean_rel_error))
                if args.txt_results != "":
                    with open(args.txt_results, 'a') as f:
                        f.write('\n')
                        f.write('AbsvalueDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(
                            AbsvalueDiff, MSE, mean_rel_error))

            if config.AttributeEstimation.calc_det_performance and config.General.experiment_path != "" and args.have_GT and config.AttributeEstimation.estimate_type == 'withKeyPoints':
                recall, precision, ap = calc_points_recall_precision_ap(T, P)
                plot_PR_curve(recall, precision, ap,
                              save_path=config.General.files_path, plots_name = 'Points_PR_curve.png')

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
        #     print('valueDiff: {} | AbsvalueDiff: {} | CountAgreement: {} | MSE {} \n'.format(
        #         valueDiff, AbsvalueDiff, CountAgreement, MSE))

        model.train()

        #if args.dataset_name == "roots":
        if  model.estimator.binary_model:
            if args.have_GT:
                return AbsvalueDiff
            else:
                return None
        else:
            if args.have_GT:
                return mean_rel_error
            else:
                return None
        # else:
        #     return CountAgreement

