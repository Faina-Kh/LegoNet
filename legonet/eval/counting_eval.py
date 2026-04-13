import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from legonet.eval.count_detection_eval import points_detection_evaluation, calc_recall_precision_ap

#from legonet.eval.both_eval_new import visualize_images
import config
import csv

from PIL import ImageFont
from PIL import ImageDraw


from thop import profile, clever_format


#
#
# def visualize_images(output, Image_name, save_path, generator, model, image):
#     # if not generator.epoch == None:
#     #     current_epoch = str(generator.epoch+1) #generator.epoch in [0,99]
#     # else:
#     #     current_epoch = 'test'
#     #
#     # visualization_path = os.path.join(save_path, 'epoch_' + current_epoch)
#
#     # if not os.path.exists(visualization_path):
#     #     os.makedirs(visualization_path)
#
#     visualization_path = save_path
#
#     fgImage_name = Image_name + "_fg.png"
#
#     # Draw GT activations:
#     #
#     background = Image.open(os.path.join(generator.base_dir, fgImage_name), 'r')
#     background = background.convert("RGBA")
#     BG_w, BG_h = background.size
#
#
#     anno = output[2][0, :, :, 0]
#     plt.figure()
#     plt.imshow(anno)
#     plt.imsave(visualization_path + '/' + Image_name + '_anno.png', anno)
#     gt_anns = Image.open(visualization_path + '/' + Image_name + '_anno.png')
#     gt_anns = gt_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
#     plt.imsave(visualization_path + '/' + Image_name + '_anno.png', gt_anns)
#     plt.close()
#
#     # out = image1 * (1.0 - alpha) + image2 * alpha
#     plt.figure()
#     plt.axis("off")
#     alphaBlended = Image.blend(gt_anns, background, 0.6)
#     plt.imshow(alphaBlended)
#     plt.imsave(visualization_path + '/' + Image_name + '_Blended_GT.png',alphaBlended )
#     plt.close()
#
#     # Relu map #######################################################################################################
#     plt.figure()
#     classification_submodel_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
#                                                           layer_name='pyramid_classification_relu')
#     classification_submodel_activations = classification_submodel_activations[0][0, :, :, 0]
#
#     plt.imshow(classification_submodel_activations)
#     plt.imsave(visualization_path + '/' + Image_name + '_Relu.png', classification_submodel_activations)
#     relu_anns = Image.open(visualization_path + '/' + Image_name + '_Relu.png')
#
#     #relu_anns = relu_anns.convert("RGBA")
#     relu_anns = relu_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
#     plt.imsave(visualization_path + '/' + Image_name + '_Relu.png', relu_anns)
#     plt.close()
#
#     plt.figure()
#     plt.axis("off")
#     alphaBlended_relu = Image.blend(relu_anns, background, 0.6)
#     plt.imshow(alphaBlended_relu)
#     plt.imsave(visualization_path + '/' + Image_name + '_Blended_Relu.png', alphaBlended_relu)
#     plt.close()
#
#     # softmax map #####################################################################################################
#
#     plt.figure()
#     local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
#                                                  layer_name='LocalSoftMax')
#     local_soft_max_activations = local_soft_max_activations[0][0, :, :, 0]
#
#     plt.imshow(local_soft_max_activations)
#     plt.imsave(visualization_path + '/' + Image_name + '_softmax.png', local_soft_max_activations)
#     softmax_anns = Image.open(visualization_path + '/' + Image_name + '_softmax.png')
#
#     #softmax_anns = softmax_anns.convert("RGBA")
#     softmax_anns = softmax_anns.resize((BG_w, BG_h))  # Image.
#     plt.imsave(visualization_path + '/' + Image_name + '_softmax.png', softmax_anns)
#     plt.close()
#
#     plt.figure()
#     plt.axis("off")
#     alphaBlended_softmax = Image.blend(softmax_anns, background, 0.6)
#     plt.imshow(alphaBlended_softmax)
#     plt.imsave(visualization_path + '/' + Image_name + '_Blended_softmax.png', alphaBlended_softmax)
#     plt.close()


def plot_RP_curve(recall, precision, ap, save_path):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.99)
    plt.fill_between(recall, precision, step='post', color='b', alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_path = os.path.join(save_path + '\\RP_curve.png')
    plt.savefig(plot_path)
    plt.close(plot_path)


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


def visualize_images(predicted_maps, gt_maps, image_name, imgToVis, draw_path, count_pred, count_GT=None):
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("arial.ttf", 60) #15) #60
    attribute = "TRL" #"count"


    if gt_maps is not None:
        if torch.sum(gt_maps[0][0]) == torch.zeros(1):
            draw_path = os.path.join(draw_path, "no roots pred" )
            os.makedirs(draw_path, exist_ok=True)

    # Draw GT activations:
    img_copy = imgToVis.copy()
    background = img_copy.convert("RGBA")
    BG_w, BG_h = background.size

    background_2 = img_copy.convert('L')   #convert image to monochrome
    background_2.save(draw_path + '/' + image_name + '_background.png')
    background_2 = Image.open(draw_path + '/' + image_name + '_background.png')
    background_2=background_2.convert("RGBA")

    if gt_maps is not None:
        anno = gt_maps.copy() #.cpu().numpy().copy()

    for i in [0]: #range(5):
        if gt_maps is not None:
            plt.imsave(draw_path + '/' + image_name + '_map_'+str(i+1)+ '_anno.png', anno[i][0])
            gt_anns = Image.open(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_anno.png')
            gt_anns = gt_anns.resize((BG_w, BG_h), Image.Resampling.LANCZOS) #Image.ANTIALIAS)
            gt_anns.save(draw_path + '/' + image_name + '_map_'+str(i+1) + '_anno.png')

            alphaBlended = Image.blend(gt_anns, background_2, 0.7)

            draw = ImageDraw.Draw(alphaBlended)

            if count_GT is not None:
                draw.text((50, 50), "GT "+ attribute+ " = "+str(np.round(count_GT, 2)), (255, 255, 255), font=font)

            alphaBlended.save(draw_path + '/' + image_name + '_map_'+str(i+1)+'_Blended_GT.png')

        # Relu map #######################################################################################################

        plt.imsave(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_Relu.png', predicted_maps[i][0].cpu())
        relu_pred = Image.open(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_Relu.png')

        relu_pred = relu_pred.resize((BG_w, BG_h))  # Image.ANTIALIAS
        relu_pred.save(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_Relu.png')

        alphaBlended_relu = Image.blend(relu_pred, background_2.convert('RGBA'), 0.7)

        draw = ImageDraw.Draw(alphaBlended_relu)
        draw.text((50, 50), "Pred "+attribute+ " = " + str(np.round(count_pred, 2)), (255, 255, 255), font=font)

        alphaBlended_relu.save(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_Blended_Relu.png')

        os.remove(draw_path + '/' + image_name + '_map_' + str(i+1) + '_Relu.png')
        if gt_maps is not None:
             os.remove(draw_path + '/' + image_name +'_map_'+str(i+1)+ '_anno.png')
    os.remove(draw_path + '/' + image_name + '_background.png')


def eval(dataloader, dataset, model, args):

    print('\n',"Start evaluation")

    model.eval()

    with torch.no_grad():

        all_GT_counts = []
        all_predicted_counts = []
        alpha = 0.1
        T, P = [], []

        all_rel_error = []

        if args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2":
            count_error_all = []
            TRL_error_all = []
            dia_mean_error_all = []
            dia_std_error_all = []


        predicted_maps = None

        if args.dataset_name == "roots":
            if args.txt_results != "":
                if args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2":
                    with open(args.txt_results, 'a') as f:
                        f.write('image| count GT | count model | count error | '
                                'TRL GT | TRL model| TRL error |'
                                'dia mean GT | dia mean model | dia mean error|'
                                'dia std GT | dia std model | dia std error |'
                                'avg rel error')
                        f.write('\n')
                else:
                    with open(args.txt_results, 'a') as f:
                        if args.have_GT:
                            f.write('image| GT | predicted | abs diff | rel_error')
                        else:
                            f.write('image| predicted')
                        f.write('\n')

            for iter_num, data in enumerate(dataloader):

                full_rgbImage_name = dataset.bgr_images_names[dataloader.batch_sampler.groups[iter_num][0]] #[iter_num]
                Image_name = full_rgbImage_name.split(".jpg")[0]  # ("_rgb")[0]


                if args.network_type == "counting_reg":
                    if args.have_GT:
                        if data['annot'][0].numpy().shape==(1,1):
                            count_GT = data['annot'][0].numpy()[0, 0]
                        else:
                            count_GT = data['annot'][0].numpy()[0,0,0]

                        count_pred = float(model([data['img'].to(config.General.device).float(), data['annot']])[0].cpu().detach().numpy())

                    else:
                        count_pred = float(model([data['img'].to(config.General.device).float()])[0].cpu().detach().numpy())

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
                        # if iter_num == 0:
                        #     print("Get FLOPS for counting_lean:")
                        #     # Use thop to profile the model
                        #     input = [data['img'].to(config.General.device).float(), data['annot']]
                        #     flops, params = profile(model, inputs=(input,))
                        #
                        #     # Print the estimated FLOPS and parameters
                        #     flops_str, params_str = clever_format([flops, params], "%.3f")
                        #     print(f"FLOPS: {flops_str}")
                        #     print(f"Params: {params_str}")

                        ###########################################################################################################




                    else:
                        count_outputs = model([data['img'].to(config.General.device).float()])


                    count_pred = float(count_outputs[0].cpu().detach().numpy())
                    #if config.General.binary_model:
                        #count_pred=np.round(count_pred)


                    # get only the prediction maps
                    predicted_maps = count_outputs[1:5]
                    predicted_maps.append(count_outputs[6])

                elif args.network_type ==  "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2":

                    if args.have_GT:
                        if args.val_csv_leaf_location_file == "":
                            count_GT = data['annot'][0].numpy()[0, 0, 0]
                        else:
                            count_GT = data['annot'][0].numpy()[0][0]

                    count_outputs = model([data['img'].to(config.General.device).float(), data['annot']])
                    if args.network_type == "counting_lean_multiple_out_V2":
                        count_pred = torch.cat(count_outputs[0]).cpu().detach().numpy()
                        count_pred = [pred[0] for pred in count_pred]
                    else:
                        count_pred = count_outputs[0].cpu().detach().numpy()[0]



                    # get only the prediction maps
                    predicted_maps = count_outputs[1:5]
                    predicted_maps.append(count_outputs[6])




                if (args.network_type == 'counting_lean' or args.network_type ==  "counting_lean_multiple_out" or
                    args.network_type ==  "counting_lean_multiple_out_V2") and args.visualize_im:
                    img = Image.open(os.path.join(dataset.base_dir, full_rgbImage_name))
                    if args.have_GT and args.val_csv_leaf_location_file != "":
                        visualize_images(predicted_maps, data['annot'][1:6], Image_name, img, args.save_img_path, count_pred, count_GT)
                    else:
                        if not args.have_GT:
                            count_GT = None
                        visualize_images(predicted_maps, None, Image_name, img, args.save_img_path, count_pred, count_GT)

                if args.have_GT:
                    all_GT_counts.append(count_GT)

                # if args.calc_det_performance:
                #     t, p = detection_evaluation(os.path.join(dataset.base_dir, Image_name), model, image,
                #                                 count_outputs[-1][0, :, :, 0], alpha)
                #     T = T + t
                #     P = P + p
                # names = ["img008_103_2021-12-14_01-50-27", "img015_103_2021-11-19_01-51-24" ]
                # if Image_name in names:
                #     a=1
                if predicted_maps is not None and args.have_GT: #and Image_name not in names:
                    if config.AttributeEstimation.calc_det_performance and dataset.csv_leaf_location_file != "":
                        if torch.sum(data['annot'][1]).item()> 0: # check that there are gt points
                            true_maps = data['annot'][1:]
                            for m in range(len(predicted_maps)):
                                for b in range(predicted_maps[0].shape[0]):
                                    t, p = points_detection_evaluation(predicted_maps[m][b], true_maps[m][b])
                                    T = T + t
                                    P = P + p

                if args.have_GT:
                    if args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2":
                        if count_GT[0] > 0:
                            rel_error = []
                            for i in range(len(count_GT)):
                                if count_GT[i] >0: # dia std can be zero even if there are roots
                                    rel_error.append(abs(count_GT[i] - count_pred[i]) / count_GT[i])

                            avg_rel_error=np.mean(rel_error)
                            all_rel_error.append(avg_rel_error)
                        else:
                            avg_rel_error = -1

                    else:
                        if not config.General.binary_model:
                            if count_GT>0:
                                rel_error = abs(count_GT - count_pred) / count_GT
                                all_rel_error.append(rel_error)
                            else:
                                rel_error = -1


                all_predicted_counts.append(count_pred)

                if args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2":
                    print('image:', Image_name)
                    if args.have_GT:
                        print(
                            'count_GT: {:.3f} | count_model: {:.3f} | TRL_GT: {:.3f} |TRL_model: {:.3f} | '
                            'dia_mean_GT: {:.3f}  | dia_mean_model: {:.3f}  | dia_std_GT: {:.3f}  | dia_std_model: {:.3f}'.format(
                                count_GT[0], np.round(count_pred[0]),
                                count_GT[1], count_pred[1],
                                count_GT[2], count_pred[2],
                                count_GT[3], count_pred[3]
                                ))
                    else:
                        print('count_model: {:.3f} | TRL_model: {:.3f} | dia_mean_model: {:.3f} | dia_std_model: {:.3f}'.format(
                            np.round(count_pred[0]),count_pred[1],count_pred[2],count_pred[3]
                        ))

                    if args.have_GT:
                        count_error = abs(count_GT[0] - np.round(count_pred[0]))/count_GT[0]
                        TRL_error = abs(count_GT[1] - count_pred[1]) / count_GT[1]
                        dia_mean_error = abs(count_GT[2] - count_pred[2])/count_GT[2]
                        if count_GT[0] > 0:
                            count_error_all.append((count_error))
                            TRL_error_all.append(TRL_error)
                            dia_mean_error_all.append(dia_mean_error)
                            if count_GT[3] > 0: # std can be zero with one root
                                dia_std_error = abs(count_GT[3] - count_pred[3]) / count_GT[3]
                                dia_std_error_all.append(dia_std_error)
                            else:
                                dia_std_error = -1


                        else:
                            count_error = -1
                            TRL_error = -1
                            dia_mean_error = -1
                            dia_std_error = -1



                        print(
                            'count_error: {:.3f} | TRL_error: {:.3f} | dia_mean_error: {:.3f}  | dia_std_error: {:.3f} | avg_rel_error: {:.3f} \n'.format(
                                count_error,
                                TRL_error,
                                dia_mean_error,
                                dia_std_error,
                                avg_rel_error))

                    if args.txt_results != "":
                        if args.have_GT:
                            with open(args.txt_results, 'a') as f:
                                f.write('{} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                                        Image_name,
                                        count_GT[0], np.round(count_pred[0]), count_error,
                                        count_GT[1], count_pred[1], TRL_error,
                                        count_GT[2], count_pred[2], dia_mean_error,
                                        count_GT[3], count_pred[3], dia_std_error,
                                        avg_rel_error))

                                f.write('\n')

                        else:
                            with open(args.txt_results, 'a') as f:
                                f.write(
                                    '{} | {} | {:.3f} | {:.3f} | {:.3f} '.format(
                                        Image_name,
                                        np.round(count_pred[0]),
                                        count_pred[1],
                                        count_pred[2],
                                        count_pred[3],
                                        ))

                                f.write('\n')


                else:
                    if config.General.binary_model:
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


        # if args.calc_det_performance:
        #     recall, precision, ap = calc_recall_precision_ap(T, P)
        #     plot_RP_curve(recall, precision, ap) #, save_path)

        print('\n Summary:')

        if args.network_type != "counting_lean_multiple_out" and args.network_type != "counting_lean_multiple_out_V2":
            if args.have_GT:
                num_of_images = len(all_GT_counts)
                CountDiff = SumOfDifferences(all_GT_counts, all_predicted_counts) / num_of_images
                AbsCountDiff = SumOfAbsDifferences(all_GT_counts, all_predicted_counts) / num_of_images
                # Rsq = np.cumsum(np.array(all_GT_counts) - np.array(all_predicted_counts))

        if args.dataset_name == "roots":

            if args.have_GT:

                if config.General.binary_model:
                    print('AbsCountDiff: {:.3f} | accuracy {:.3f} \n'.format(
                        AbsCountDiff, 1-AbsCountDiff))

                    if args.txt_results != "":
                        with open(args.txt_results, 'a') as f:
                            f.write('\n')
                            f.write('AbsCountDiff: {:.3f} | accuracy {:.3f} \n'.format(
                                AbsCountDiff, 1-AbsCountDiff ))

                else:

                    mean_rel_error = np.mean(all_rel_error)

                    if args.network_type != "counting_lean_multiple_out" and args.network_type != "counting_lean_multiple_out_V2":
                        SE = 0
                        count_non_zero = 0
                        for i in range(len(all_GT_counts)):
                            if all_GT_counts[i] >0:
                                count_non_zero+=1
                                SE += (all_GT_counts[i]-all_predicted_counts[i])** 2
                        if count_non_zero>0:
                            MSE = SE/count_non_zero
                        else:
                            MSE = None

                        print('AbsCountDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(AbsCountDiff, MSE, mean_rel_error))

                        if args.txt_results != "":
                            with open(args.txt_results, 'a') as f:
                                f.write('\n')
                                f.write('AbsCountDiff: {:.3f} | MSE {:.3f} | RelError (gt>0): {:.3f} \n'.format(
                                    AbsCountDiff, MSE, mean_rel_error))


                if config.AttributeEstimation.calc_det_performance and args.save_detection_eval_path != "" and args.have_GT:
                    recall, precision, ap = calc_recall_precision_ap(T, P)
                    plot_RP_curve(recall, precision, ap,
                                  save_path=args.save_detection_eval_path)  # config.General.experiment_path)

                    # print recall and precision  to csv
                    csv_columns = ['recall', 'precision']
                    csv_file = os.path.join(args.save_detection_eval_path, "parts_recall_precision.csv")
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
            if args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2" and args.have_GT:
                print("avg count_error:", np.mean(count_error_all))
                print("avg TRL_error:", np.mean(TRL_error_all))
                print("avg dia_mean_error:", np.mean(dia_mean_error_all))
                print("avg dia_std_error:", np.mean(dia_std_error_all))

            if config.General.binary_model:
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

