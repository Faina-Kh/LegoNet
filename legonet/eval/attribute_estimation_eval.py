import torch
import os
import numpy as np
import config
import csv
from PIL import Image
from thop import profile, clever_format

from legonet.eval.KP_detection_eval import points_detection_t_p, calc_points_recall_precision_ap, visualize_KeyPointsHeatmaps
from legonet.eval.detection_eval import plot_PR_curve




def SumOfDifferences(A, B):
    # calculate sum of  A - B
    #  A and B must have the same size
    out = 0
    for i in range(len(A)):
        out+=(A[i]-B[i])
    return out


def SumOfAbsDifferences(A,B):
    # calculate sum of A-B
    # A and B must have the same size

    out = 0
    for i in range(len(A)):
        out += abs(A[i] - B[i])

    return out


def eval(dataloader, dataset, model, args, do_profile = False):

    print("Start evaluation")

    model.eval()

    with (torch.no_grad()):

        all_GT_counts = []
        all_predicted_counts = []
        alpha = 0.1
        T, P = [], []

        all_rel_error = []

        predicted_maps = None

        if args.dataset_name == "roots":
            if args.txt_results != "":

                with open(args.txt_results, 'a') as f:
                    if args.have_GT:
                        f.write('image| GT | predicted | abs diff | rel_error')
                    else:
                        f.write('image| predicted')
                    f.write('\n')

            for iter_num, data in enumerate(dataloader):

                full_rgbImage_name = dataset.bgr_images_names[dataloader.batch_sampler.groups[iter_num][0]] #[iter_num]

                if full_rgbImage_name.lower().endswith((".jpg", ".jpeg")):
                    Image_name = full_rgbImage_name.split(".jpg")[0]
                elif full_rgbImage_name.lower().endswith(".png"):
                    Image_name = full_rgbImage_name.split(".png")[0]

                #Image_name = full_rgbImage_name.split(".jpg")[0]  # ("_rgb")[0]

                if args.network_type == "counting_reg":
                    if args.have_GT:
                        if data['annot'][0].numpy().shape==(1,1):
                            count_GT = data['annot'][0].numpy()[0, 0]
                        else:
                            count_GT = data['annot'][0].numpy()[0,0,0]

                        count_pred = float(model([data['img'].to(config.General.device).float(), data['annot']])[0].squeeze().item())

                    else:
                        count_pred = float(model([data['img'].to(config.General.device).float()])[0].squeeze().item())

                    if count_pred<0:
                        count_pred=0

                elif args.network_type == "counting_lean":
                    if args.have_GT:
                        if args.val_csv_leaf_location_file == "":
                            count_GT = data['annot'][0].numpy()[0, 0, 0]
                        else:
                            count_GT = data['annot'][0].numpy()[0][0]

                        #print(Image_name,": ", count_GT)

                        count_outputs = model([data['img'].to(config.General.device).float(), data['annot']])

                        ###########################################################################################################
                        if iter_num == 0 and do_profile:
                            print("Get FLOPS for counting_lean:")
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


                    count_pred = count_outputs[0].squeeze().item() #float(count_outputs[0].cpu().detach().numpy())
                    #if config.General.binary_model:
                        #count_pred=np.round(count_pred)


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

                if args.network_type == 'counting_lean' and config.General.to_draw:
                    img = Image.open(os.path.join(dataset.base_dir, full_rgbImage_name))
                    if args.have_GT and args.val_csv_leaf_location_file != "":
                        gt_maps = data['annot'][1:6]

                    else:
                        count_GT = None
                        gt_maps = None

                    for i in [4]:  # range(5): #which heatmaps to visualize, numbered from 0 to 4
                        map_name = Image_name + '_map_' + str(i + 1)
                        predicted_map = predicted_maps[i].cpu().numpy()[0]
                        if gt_maps is not None:
                            gt_map = gt_maps[i].cpu().numpy()[0]
                            visualize_KeyPointsHeatmaps(predicted_map, gt_map, Image_name, map_name, img,
                                                        config.DrawProperties.maps_path, count_pred, count_GT)
                        else:
                            visualize_KeyPointsHeatmaps(predicted_map, None, Image_name, map_name, img,
                                                        config.DrawProperties.maps_path, count_pred, count_GT)


                if args.have_GT:
                    all_GT_counts.append(count_GT)

                if predicted_maps is not None and args.have_GT: #and Image_name not in names:
                    if config.AttributeEstimation.calc_det_performance and dataset.csv_leaf_location_file != "":
                        if torch.sum(data['annot'][1]).item()> 0: # check that there are gt points
                            true_maps = data['annot'][1:]
                            for m in range(len(predicted_maps)):
                                for b in range(predicted_maps[0].shape[0]):
                                    t, p = points_detection_t_p(predicted_maps[m][b], true_maps[m][b])
                                    T = T + t
                                    P = P + p

                if args.have_GT:
                    if not model.estimator.binary_model: #config.General.binary_model:
                        if count_GT>0:
                            rel_error = abs(count_GT - count_pred) / count_GT
                            all_rel_error.append(rel_error)
                        else:
                            rel_error = -1


                all_predicted_counts.append(count_pred)

                if  model.estimator.binary_model:
                    if args.have_GT:

                        print(
                            'image: {} | GT: {:.3f} | predicted: {:.3f}'.format(
                                Image_name, count_GT, count_pred,
                                ))

                        if args.txt_results != "":
                            with open(args.txt_results, 'a') as f:
                                f.write('{} | {:.3f} | {:.3f} '.format(
                                    Image_name, count_GT, count_pred
                                    ))
                                f.write('\n')

                    else:

                        print(
                            'image: {} | predicted: {:.3f}'.format(
                                Image_name, count_pred))

                        if args.txt_results != "":
                            with open(args.txt_results, 'a') as f:
                                f.write('{} | {:.3f} '.format(
                                    Image_name, count_pred
                                ))
                                f.write('\n')

                else:
                    if args.have_GT:
                        print(
                            'image: {} | GT: {:.3f} | predicted: {:.3f} | abs diff: {:.3f} | rel_error: {:.3f}'.format(
                                Image_name, count_GT, count_pred,
                                abs(count_GT - count_pred), rel_error))

                        if args.txt_results != "":
                            with open(args.txt_results, 'a') as f:
                                f.write('{} | {:.3f} | {:.3f} | {:.3f} |{:.3f}'.format(
                                        Image_name, count_GT, count_pred,
                                        abs(count_GT - count_pred), rel_error))
                                f.write('\n')
                    else:
                        print(
                            'image: {} | predicted: {:.3f} '.format(
                                Image_name, count_pred))

                        if args.txt_results != "":
                            with open(args.txt_results, 'a') as f:
                                f.write('{} | {:.3f} |'.format(
                                    Image_name, count_pred ))
                                f.write('\n')



        else:

            for index in range(len(dataset)):
                data = dataset[index]
                count_GT = data['annot'][0][0]

                # run network
                image = torch.tensor(data['img']).permute(2, 0, 1)
                count_outputs = model(image.to(config.General.device).float().unsqueeze(dim=0))

                full_rgbImage_name = dataset.bgr_images_names[index]
                Image_name = full_rgbImage_name.split("_rgb")[0]

                if len(count_outputs):
                    count_pred = count_outputs[0]
                    count_pred = count_pred.cpu().item()

                    full_rgbImage_name = dataset.bgr_images_names[index]
                    Image_name = full_rgbImage_name.split("_rgb")[0]

                    # if visualize_im:
                    #     if not generator.epoch == None:
                    #         if generator.epoch==0 or (generator.epoch+1) % 20 == 0 :
                    #             visualize_images(output, Image_name, save_path, generator, model, image)
                    #     else:
                    #         visualize_images(output, Image_name, save_path, generator, model, image)

                    all_GT_counts.append(count_GT)

                    # if args.calc_det_performance:
                    #     t, p = detection_evaluation(os.path.join(dataset.base_dir, Image_name), model, image,
                    #                                 count_outputs[-1][0, :, :, 0], alpha)
                    #     T = T + t
                    #     P = P + p

                    all_predicted_counts.append(np.round(count_pred))
                    print(
                        'image: {} | GT: {} | predicted: {} ({}) | abs diff: {}'.format(
                            Image_name, int(count_GT), np.round(count_pred), count_pred,
                            abs(count_GT - count_pred)))


        print('\n Summary:')

        if args.have_GT:
            num_of_images = len(all_GT_counts)
            CountDiff = SumOfDifferences(all_GT_counts, all_predicted_counts) / num_of_images
            AbsCountDiff = SumOfAbsDifferences(all_GT_counts, all_predicted_counts) / num_of_images
            # Rsq = np.cumsum(np.array(all_GT_counts) - np.array(all_predicted_counts))

        if args.dataset_name == "roots":

            if args.have_GT:

                if  model.estimator.binary_model:
                    print('AbsCountDiff: {:.3f} | accuracy {:.3f} \n'.format(
                        AbsCountDiff, 1-AbsCountDiff))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('\n')
                            f.write('AbsCountDiff: {:.3f} | accuracy {:.3f} \n'.format(
                                AbsCountDiff, 1-AbsCountDiff ))

                else:

                    mean_rel_error = np.mean(all_rel_error)

                    SE = 0
                    count_non_zero = 0
                    for i in range(len(all_GT_counts)):
                        if all_GT_counts[i] > 0:
                            count_non_zero += 1
                            SE += (all_GT_counts[i] - all_predicted_counts[i]) ** 2
                    if count_non_zero > 0:
                        MSE = SE / count_non_zero
                    else:
                        MSE = None

                    print('AbsCountDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(AbsCountDiff, MSE,
                                                                                                  mean_rel_error))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('\n')
                            f.write('AbsCountDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(
                                AbsCountDiff, MSE, mean_rel_error))


                if config.AttributeEstimation.calc_det_performance and config.General.experiment_path != "" and args.have_GT and config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    recall, precision, ap = calc_points_recall_precision_ap(T, P)
                    plot_PR_curve(recall, precision, ap,
                                  save_path=config.General.files_path, plots_name = 'Points_PR_curve.png')  # config.General.experiment_path)

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

        else:
            MSE = np.mean((np.array(all_GT_counts) - np.array(all_predicted_counts)) ** 2)
            countAgr = 0
            for i in range(num_of_images):
                if all_GT_counts[i] == all_predicted_counts[i]:
                    countAgr += 1
            CountAgreement = countAgr / num_of_images

            # R_2 = r2_score(all_GT_counts, all_predicted_counts)
            # TODO - change r2 score function
            R_2 = 0

            print('CountDiff: {} | AbsCountDiff: {} | CountAgreement: {} | MSE {} \n'.format(
                CountDiff, AbsCountDiff, CountAgreement, MSE))

        model.train()

        if args.dataset_name == "roots":

            if  model.estimator.binary_model:
                if args.have_GT:
                    return AbsCountDiff
                else:
                    return None

            else:
                if args.have_GT:
                    return mean_rel_error
                else:
                    return None

        else:
            return CountAgreement

