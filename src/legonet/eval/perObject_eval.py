"""Per-object detection and attribute-estimation evaluation utilities.

The public :func:`eval` entry point is retained for compatibility with the
training and inference pipelines.  Small private helpers keep matching rules
independently testable while the legacy evaluation routine is decomposed.
"""

import os
import matplotlib
import numpy as np
import PIL
import torch

matplotlib.use("TkAgg")
from PIL import Image
from thop import profile, clever_format

from legonet import config
from legonet.eval.attribute_estimation_eval import SumOfAbsDifferences
from legonet.eval.evaluation_finalization import finalize_evaluation
from legonet.eval.image_context import prepare_image_context
from legonet.eval.model_input import build_per_object_model_input
from legonet.eval.prediction_summary import record_per_image_predictions
from legonet.eval.ground_truth_preparation import (
    prepare_image_ground_truth,
    split_boxes_by_annotations as _prepare_gt_boxes_for_attribute_eval,
)
from legonet.eval.evaluation_state import initiate_global_dicts
from legonet.eval.detection_bookkeeping import record_detection_bookkeeping
from legonet.eval.KP_detection_eval import points_detection_t_p
from legonet.eval.matching import (
    assign_detection_to_gt as _assign_detection_to_gt,
    choose_boxes_by_IoUandPrc,
    match_detections_to_gt as _match_detections_to_gt,
)
from legonet.eval.classification_metrics import decode_class_predictions
from legonet.eval.metric_aggregation import (
    aggregate_post_loop_metrics,
    compute_positive_crop_count_metrics as _compute_positive_crop_count_metrics,
)
from legonet.eval.per_object_result import ClassificationType
from legonet.eval.visualization import (
    save_keypoint_heatmap,
    save_object_visualizations,
)
from legonet.my_dataloader import UnNormalizer
from legonet.progress import print_image_progress
from legonet.utils import printf



unnormalize = UnNormalizer()


def _get_detections(detection_outputs, scale):
    scores, labels, boxes = detection_outputs
    boxes = boxes.cpu().numpy()

    # correct boxes for image scale
    boxes /= scale

    return scores.cpu().numpy(), boxes

def _get_count_and_box_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_box_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]
    all_count_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    total_images = len(generator)
    for i in range(total_images):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            box_anns=annotations[0]  #[x1,y1,x2,y2,class,box_id]
            if len(annotations[1])>0:
                count_anns=np.array(annotations[1][0])
                all_count_annotations[i][label] = count_anns[count_anns[:, 1] == label, :].copy()

            else:
                count_anns=[]
                all_count_annotations[i][label] = []

            if len(box_anns) > 0:
                all_box_annotations[i][label] = box_anns[box_anns[:, 4] == label, :].copy()
            else:
                all_box_annotations[i][label] = []


        print_image_progress(
            "Loading per-object annotations:",
            i + 1,
            total_images,
        )

    if total_images == 0:
        print_image_progress("Loading per-object annotations:", 0, 0)
    return all_box_annotations,all_count_annotations



def _geometric_point_centers_map(
    point_annotations,
    crop_box,
    image_scale,
    output_shape,
    crop_size,
):
    """Project points geometrically inside a crop onto a binary center map.

    This helper is used only for point-detection metrics. It deliberately
    ignores GT bbox identifiers so roots and grapes use the same geometric
    point-to-crop association without changing model inputs or attribute
    calculations.
    """
    scale = float(np.asarray(image_scale).reshape(-1)[0])
    x1, y1, x2, y2 = (float(value) for value in crop_box[:4])
    output_height, output_width = (int(value) for value in output_shape)
    crop_height, crop_width = (float(value) for value in crop_size)
    center_map = np.zeros((output_height, output_width), dtype=np.float32)

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return center_map

    for point in point_annotations:
        point_x = float(point["x"]) * scale
        point_y = float(point["y"]) * scale
        if not (x1 <= point_x <= x2 and y1 <= point_y <= y2):
            continue

        crop_x = (point_x - x1) * crop_width / box_width
        crop_y = (point_y - y1) * crop_height / box_height
        map_x = int(round(crop_x * output_width / crop_width))
        map_y = int(round(crop_y * output_height / crop_height))
        map_x = min(max(map_x, 0), output_width - 1)
        map_y = min(max(map_y, 0), output_height - 1)
        center_map[map_y, map_x] = 1.0

    return center_map



def eval(dataset, dataloader, sampler, model, verbose=True, to_draw=True, draw_path= "",
         print_to_files=False, args = None, do_profile = False,
         detection_metrics=None, evaluate_points=False):
    is_roots_2 = (
                config.Detect_and_Estimate.type == "per_object_attributes" or config.Detect_and_Estimate.type == "per_object_attributes_multibranch")

    model.eval()

    if to_draw:
        crops_path = os.path.join(draw_path, "crops")
        if not os.path.exists(crops_path):
            os.makedirs(crops_path)

        gt_path = os.path.join(draw_path, "gt only")
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)

    # gather all annotations, per image, per label
    if args.have_GT:
        all_box_annotations, all_count_annotations = _get_count_and_box_annotations(dataset)

    with (((((((((((torch.no_grad()))))))))))):

        state = initiate_global_dicts(initiate=True)
        print()
        for iter_num, data in enumerate(dataloader):

            # get per image stats
            crops_count_GT = []
            orig_count_GT = []
            count_pred = []

            if is_roots_2:
                orig_TRL_GT = []
                TRL_pred = []
                orig_dia_GT = []
                dia_pred = []
                color_pred = []
                orig_color_GT = []

            image_context = prepare_image_context(
                dataset,
                sampler,
                iter_num,
                data,
                have_gt=args.have_GT,
                all_box_annotations=(
                    all_box_annotations if args.have_GT else None
                ),
                all_count_annotations=(
                    all_count_annotations if args.have_GT else None
                ),
            )
            image = image_context.image
            scale = image_context.scale
            group_idx = image_context.group_indices
            image_name = image_context.image_name
            image_path = image_context.image_path
            box_annotations_temp = image_context.box_annotations
            gt_counts_temp = image_context.count_annotations

            state = initiate_global_dicts(state, image_name)
            if verbose:
                printf("##############################################################################################\n")
                printf("image: %s\n", image_name)
                printf("##############################################################################################\n")

            prepared_gt = prepare_image_ground_truth(
                state,
                image_name,
                box_annotations_temp,
                gt_counts_temp,
                have_gt=args.have_GT,
                attributes=is_roots_2,
            )
            im_gt_avg = prepared_gt.image_gt_average
            TRL_im_gt_sum = prepared_gt.image_trl_sum
            dia_im_gt_avg = prepared_gt.image_diameter_average
            box_annotations_all = prepared_gt.all_boxes
            box_annotations_withPoints = prepared_gt.annotated_boxes
            gt_counts = prepared_gt.matched_counts

            if not prepared_gt.include_image:
                if verbose:
                    if prepared_gt.skip_reason == "no_gt_boxes":
                        printf("No gt boxes ...\n")
                    else:
                        printf("Has gt boxes but no gt points in any gt box...\n")
                    printf("Skipping this image for per-object evaluation...\n")
                    print()
                continue

            model_input = build_per_object_model_input(
                image,
                data,
                group_idx,
                have_gt=args.have_GT,
                device=config.General.device,
            )
            (
                detection_outputs,
                estimation_outputs,
                sample_anns,
                relevant_points_anns,
                crops_orig_boxes,
            ) = model(model_input)

            if iter_num==0 and do_profile:
                print("Both_2 FLOPS:")

                # Use thop to profile the model
                flops, params = profile(model, inputs=(model_input,))

                # Print the estimated FLOPS and parameters
                flops_str, params_str = clever_format([flops, params], "%.3f")
                print(f"FLOPS: {flops_str}")
                print(f"Params: {params_str}")

            record_per_image_predictions(
                state,
                image_name,
                estimation_outputs,
                attributes=is_roots_2,
            )

            # detection_outputs - outputs of the detection part (based on module where), after filtering by nms and min score
            # estimation_outputs - prediction of counting per box from detection_outputs
            # sample_anns - has the crop in its 'img' key. In training, it has also 'points_annot' key that holds the gt annotations per crop - relevant
            # for evaluation during training.
            ###################################################################################################################################################

            # original image
            orig_img = Image.open(image_path)

            # bbox_pred - all predicted boxes - with or without points in it - rescaled to the orig img size

            obj_scores, bbox_pred = _get_detections(detection_outputs, scale) #bbox_pred coordinates for original image size

            if bbox_pred is None or len(bbox_pred)==0:
                if verbose:
                    printf("Image has no predicted boxes...\n")
                    print()

                if not image_name in state['no_predictions'].keys():

                    state['no_predictions'][image_name]= {
                   'gt_count': [], 'label': [], 'score': [], 'max_overlap': []
                    }

                    if config.Detect_and_Estimate.type == "per_object_counting":
                        state['no_predictions'][image_name].update({'pred': []})

                    if is_roots_2:
                        state['no_predictions'][image_name].update({
                            'TRL_pred': [], 'TRL_gt': [], 'dia_pred': [], 'dia_gt': [], 'color_pred': [], 'color_gt': []
                        })

                if config.Detect_and_Estimate.type == "per_object_counting":
                    state['no_predictions'][image_name]['pred'].append(0)

                state['no_predictions'][image_name]['gt_count'].append(im_gt_avg)
                state['no_predictions'][image_name]['label'].append(-1)
                state['no_predictions'][image_name]['score'].append(-1)
                state['no_predictions'][image_name]['max_overlap'].append(-1)

                if is_roots_2:
                    state['no_predictions'][image_name]['TRL_pred'].append(0)
                    state['no_predictions'][image_name]['TRL_gt'].append(TRL_im_gt_sum)
                    state['no_predictions'][image_name]['dia_pred'].append(0)
                    state['no_predictions'][image_name]['dia_gt'].append(dia_im_gt_avg)
                    state['no_predictions'][image_name]['color_pred'].append(-1)
                    if TRL_im_gt_sum==0:
                        state['no_predictions'][image_name]['color_gt'].append(-1)
                    else:
                        state['no_predictions'][image_name]['color_gt'].append(im_gt_avg)  # no gt avg color

                continue

            # Detection evaluation
            # add scores to pred boxes
            detections = np.concatenate((bbox_pred, np.array([obj_scores]).T), axis=1)
            state['all_detections'].append(detections)

            if args.have_GT:
                if len(dataset.image_data_points_location[image_name]) > 0:
                    point_anns = dataset.image_data_points_location[image_name]  # data from kcsv file (original coords)
                else:
                    dataset.image_data_points_location[image_name] = []
                current_anns = []
                if len(box_annotations_all) > 0:
                    gt_boxes = data['bbox_annot'][0]

                    for i in range(gt_boxes.shape[0]): # get coordinates of orig image
                        x1 = gt_boxes[i, 0].numpy() / scale
                        y1 = gt_boxes[i, 1].numpy() / scale
                        x2 = gt_boxes[i, 2].numpy() / scale
                        y2 = gt_boxes[i, 3].numpy() / scale

                        current_anns.append(np.array([x1,y1,x2,y2]).reshape(4))#.reshape(4))

                state['all_annotations'].append(np.array(current_anns)) #np.array([x1,y1,x2,y2]).reshape(4))

            if to_draw:
                draw = PIL.ImageDraw.Draw(orig_img)

                if args.have_GT:
                    for p in point_anns:
                        r = config.DrawProperties.POINT_RADIUS
                        draw.ellipse((p['x']-r, p['y']-r, p['x']+r, p['y']+r), fill="black",
                                     width=config.DrawProperties.LINE_WIDTH)

                    if len(box_annotations_all) > 0:
                        gt_boxes = data['bbox_annot'][0]
                        for i in range(gt_boxes.shape[0]):
                            x1 = gt_boxes[i, 0].item() / scale[0]
                            y1 = gt_boxes[i, 1].item() / scale[0]
                            x2 = gt_boxes[i, 2].item() / scale[0]
                            y2 = gt_boxes[i, 3].item()/ scale[0]
                            draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)

                            orig_img_2 = Image.open(image_path)
                            draw_gt = PIL.ImageDraw.Draw(orig_img_2)
                            draw_gt.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)
                            orig_img_2.save(os.path.join(gt_path, image_name.split(".jpg")[0]+"_gt_"+str(i) + ".jpg"))

                    orig_img.save(os.path.join(gt_path, image_name))

                if len(bbox_pred)> 0:
                    for b in range(bbox_pred.shape[0]):
                        x1 = bbox_pred[b, 0]
                        y1 = bbox_pred[b, 1]
                        x2 = bbox_pred[b, 2]
                        y2 = bbox_pred[b, 3]
                        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=config.DrawProperties.LINE_WIDTH)

                orig_img.save(os.path.join(draw_path, image_name))


            ############################################################################################################
            # Preparing the evaluation of the counting results per crop - doesn't depend on the detection performance -
            # we only need to find and evaluate the crops that include gt points - otherwise we'll compare
            # the predicted count to 0, probably since the points weren't annotated
            ############################################################################################################

            adjusted_crops_orig_boxes = []

            if sample_anns is not None: #not the detect_with_points model or no predictions
                if config.AttributeEstimation.estimate_type == 'withKeyPoints' and estimation_outputs is not None:
                    orig_predicted_detection_maps = [estimation_outputs[1], estimation_outputs[2], estimation_outputs[3],
                                                     estimation_outputs[4], estimation_outputs[6]]  # estimation_outputs[6]
                    all_predicted_detection_maps = []
                    all_predicted_detection_maps_toDraw = []

                    all_crops_GT_detections_maps = []

                if 'points_annot' not in sample_anns.keys():
                    sample_anns['points_annot']=[]

            if not isinstance(bbox_pred, list):
                for b in range(bbox_pred.shape[0]):

                    if estimation_outputs is not None:
                        if is_roots_2:
                            decoded_color = int(
                                decode_class_predictions(
                                    [estimation_outputs[0][b][0].cpu().item()],
                                    ClassificationType.BINARY,
                                )[0]
                            )
                            state['detections_data_any_crop'][image_name]['color_pred'].append(decoded_color)
                            state['detections_data_any_crop'][image_name]['TRL_pred'].append(estimation_outputs[0][b][1].cpu().item())
                            state['detections_data_any_crop'][image_name]['dia_pred'].append(estimation_outputs[0][b][2].cpu().item())

                            color_pred.append(decoded_color)
                            TRL_pred.append(estimation_outputs[0][b][1].cpu().item())
                            dia_pred.append(estimation_outputs[0][b][2].cpu().item())

                            #adjusted_crops_orig_boxes.append(crops_orig_boxes[b])
                        else:
                            prediction = np.round(estimation_outputs[0][b].cpu().item())
                            state['detections_data_any_crop'][image_name]['pred'].append(prediction)
                            count_pred.append(prediction)

                    if len(crops_orig_boxes) > 0:
                        adjusted_crops_orig_boxes.append(crops_orig_boxes[b])

                    if config.AttributeEstimation.estimate_type == 'withKeyPoints' and estimation_outputs is not None:
                        #if config.DrawProperties.DRAW_MAPS:
                        all_predicted_detection_maps.append(orig_predicted_detection_maps[-1][b])

                        #if to_draw and config.DrawProperties.DRAW_MAPS: #draw_maps:
                        all_predicted_detection_maps_toDraw.append([
                            orig_predicted_detection_maps[0][b], orig_predicted_detection_maps[1][b],
                            orig_predicted_detection_maps[2][b], orig_predicted_detection_maps[3][b],
                            orig_predicted_detection_maps[4][b]])

                        if sample_anns is not None:
                            all_crops_GT_detections_maps.append(
                                torch.tensor(
                                    _geometric_point_centers_map(
                                        dataset.image_data_points_location[
                                            image_name
                                        ],
                                        crops_orig_boxes[b],
                                        scale,
                                        all_predicted_detection_maps[
                                            -1
                                        ].shape,
                                        config.AttributeEstimation.crops_size,
                                    )
                                )
                            )

                    if sample_anns is not None and args.have_GT: #assuming having GT anns
                        current_count = sample_anns['points_annot'][0][b].item()
                        if current_count ==0: #empty crop
                            state['crops_without_gt_points'] +=1

                        crops_count_GT.append(current_count)  # just to know if it's an empty crop

            if sample_anns is not None:
                if len(crops_count_GT)>0:
                    max_gt_count_of_crop = np.max(crops_count_GT)
                else:
                    max_gt_count_of_crop = -1

                if max_gt_count_of_crop == -1 and not is_roots_2:
                    sample_anns = None
                else:
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                        if args.have_GT:
                            all_crops_GT_detections_maps = torch.cat([torch.unsqueeze(map, dim=0) for map in all_crops_GT_detections_maps], dim=0)

                        if len(all_predicted_detection_maps)>1:
                            all_predicted_detection_maps = torch.cat([torch.unsqueeze(map, dim=0) for map in all_predicted_detection_maps], dim=0)
                        else:
                            all_predicted_detection_maps = all_predicted_detection_maps[0].unsqueeze(dim=0)

            if np.sum(crops_count_GT)==0 and len(bbox_pred)>0: #sample_anns is None
                if verbose and args.have_GT:
                    printf("No gt points in any predicted crop...\n")
                    print()

            ############################################################################################################
            # Preparing the evaluation in comparison to the gt points' count of the relevant gt object -
            # the object with iou>thresh of the predicted box with the gt box
            ############################################################################################################

            if to_draw:
                if len(box_annotations_all) > 0:
                    gt_boxes = data['bbox_annot'][0]
                    for i in range(gt_boxes.shape[0]):
                        x1 = gt_boxes[i, 0].item()/scale[0]
                        y1 = gt_boxes[i, 1].item()/scale[0]
                        x2 = gt_boxes[i, 2].item()/scale[0]
                        y2 = gt_boxes[i, 3].item()/scale[0]
                        draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)

                else:
                    gt_boxes = []

                #orig_img.show(title=image_name)
                # save the original img with pred bbox and gt anns
                orig_img.save(os.path.join(draw_path,image_name))

            if len(box_annotations_withPoints)>0:
                for i in range(box_annotations_withPoints.shape[0]):
                    box_annotations_withPoints[i, 0] = box_annotations_withPoints[i, 0] * scale[0]
                    box_annotations_withPoints[i, 1] = box_annotations_withPoints[i, 1] * scale[0]
                    box_annotations_withPoints[i, 2] = box_annotations_withPoints[i, 2] * scale[0]
                    box_annotations_withPoints[i, 3] = box_annotations_withPoints[i, 3] * scale[0]

            bookkeeping = record_detection_bookkeeping(
                state=state,
                image_name=image_name,
                predicted_boxes=adjusted_crops_orig_boxes,
                annotated_boxes=box_annotations_withPoints,
                gt_values=gt_counts,
                attributes=is_roots_2,
                counting=(
                    config.Detect_and_Estimate.type == "per_object_counting"
                ),
                iou_threshold=config.Detection.iou_threshold,
            )
            orig_count_GT = bookkeeping.count_ground_truth
            orig_TRL_GT = bookkeeping.trl_ground_truth
            orig_dia_GT = bookkeeping.diameter_ground_truth
            orig_color_GT = bookkeeping.color_ground_truth
            max_overlap_array = bookkeeping.max_overlaps

            ############################################################################################################
            # Evaluate the results of the given image
            ############################################################################################################
            if estimation_outputs is not None:
                if len(estimation_outputs):
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                        if evaluate_points:
                            if len(all_predicted_detection_maps)>0:
                                for b in range(all_predicted_detection_maps.shape[0]):
                                    t, p = points_detection_t_p(all_predicted_detection_maps[b, :, :], all_crops_GT_detections_maps[b, :, :])
                                    state['T']=state['T']+ t
                                    state['P']=state['P']+ p

                    if not is_roots_2:
                        state['all_crops_GT_counts'].append(crops_count_GT)  # based on the number of points in the crop
                        state['all_predicted_counts'].append(np.round(count_pred))
                        state['all_orig_GT_counts'].append(orig_count_GT) # count in the corresponding annotated box, it is -1 if the crop is false positive

                    else:
                        #state['all_crops_GT_TRL'].append(crops_TRL_GT)
                        state['all_predicted_TRL'].append(TRL_pred)
                        state['all_orig_GT_TRL'].append(orig_TRL_GT)

                        #state['all_crops_GT_dia'].append(crops_dia_GT)
                        state['all_predicted_dia'].append(dia_pred)
                        state['all_orig_GT_dia'].append(orig_dia_GT)

                        state['all_predicted_color'].append(np.round(color_pred))
                        state['all_orig_GT_color'].append(orig_color_GT)

                    maps_idx=0
                    for i in range(len(crops_count_GT)):
                        if not is_roots_2:
                            if len(orig_count_GT) > 0:
                                if orig_count_GT[i] != -1:
                                    state['orig_abs_diff'].append(abs(orig_count_GT[i] - np.round(count_pred[i]))) #count_pred[i]))
                                    if orig_count_GT[i] > 0:
                                        state['orig_rel_error'].append(abs(orig_count_GT[i] - np.round(count_pred[i])) / orig_count_GT[i])

                        elif crops_count_GT[i] > 0:
                            if orig_color_GT[i] != -1:
                                state['orig_abs_diff_TRL'].append(abs(orig_TRL_GT[i] - TRL_pred[i])) # count_pred[i]))
                                state['orig_rel_error_TRL'].append(abs(orig_TRL_GT[i] - TRL_pred[i]) / orig_TRL_GT[i])

                                state['orig_abs_diff_dia'].append(abs(orig_dia_GT[i] - dia_pred[i]))  # count_pred[i]))
                                state['orig_rel_error_dia'].append(abs(orig_dia_GT[i] - dia_pred[i]) / orig_dia_GT[i])

                                state['orig_abs_diff_color'].append(abs(orig_color_GT[i] - color_pred[i]))  # count_pred[i]))

                        if verbose:

                            if (not is_roots_2 and len(orig_count_GT) > 0) \
                                    or (is_roots_2 and len(orig_color_GT) > 0):

                                    if not is_roots_2:
                                        if orig_count_GT[i] != -1:
                                            printf("orig_count_GT: %d | orig_predicted_count: %d |orig_abs_diff: %.3f | orig_rel_error: %.3f\n",
                                                   int(orig_count_GT[i]), np.round(count_pred[i]) ,state['orig_abs_diff'][-1], state['orig_rel_error'][-1])

                                            printf('max_overlap =  %.3f \n', max_overlap_array[i])

                                        else:
                                            print('No gt box with iou>=', config.Detection.iou_threshold)
                                            printf('max_overlap =  %.3f \n', max_overlap_array[i])

                                    else:
                                        if orig_color_GT[i] != -1:
                                            printf(
                                                "orig_color_GT: %.3f | color_pred: %.3f | color_orig_abs_diff: %.3f \n",
                                                orig_color_GT[i], color_pred[i], state['orig_abs_diff_color'][-1]
                                            )
                                            printf(
                                                "orig_TRL_GT: %.3f | TRL_pred: %.3f | TRL_orig_abs_diff: %.3f | TRL_orig_rel_error:  %.3f\n",
                                                orig_TRL_GT[i], TRL_pred[i] , state['orig_abs_diff_TRL'][-1], state['orig_rel_error_TRL'][-1]
                                            )
                                            printf(
                                                "orig_dia_GT: %.3f | dia_pred: %.3f | dia_orig_abs_diff: %.3f | dia_orig_rel_error: %.3f\n",
                                                 orig_dia_GT[i], dia_pred[i], state['orig_abs_diff_dia'][-1], state['orig_rel_error_dia'][-1]
                                            )

                                            printf('max_overlap =   %.3f \n', max_overlap_array[i])


                                        else:
                                            print('No gt box with iou>=', config.Detection.iou_threshold)
                                            if max_overlap_array is not None:
                                                printf('max_overlap =   %.3f \n', max_overlap_array[i])
                            print()

                        if to_draw: # drawings of true detections
                            if len(crops_count_GT) > 0:
                                bbox_crop = sample_anns['img'][i].clone()
                                gt_box_id = int(
                                    state['detections_data_any_crop'][image_name][
                                        'gt_box_id'
                                    ][i].item()
                                )
                                crop_image = save_object_visualizations(
                                    image_path=image_path,
                                    image_name=image_name,
                                    crop_index=i,
                                    bbox_crop=bbox_crop,
                                    predicted_box=crops_orig_boxes[i],
                                    scale=scale,
                                    gt_boxes=gt_boxes,
                                    gt_box_id=gt_box_id,
                                    roots_attributes=is_roots_2,
                                    image_points=point_anns,
                                    crop_points=relevant_points_anns[i],
                                    has_positive_target=crops_count_GT[i] != 0,
                                    draw_path=draw_path,
                                    crops_path=crops_path,
                                    line_width=config.DrawProperties.LINE_WIDTH,
                                    unnormalize=unnormalize,
                                )

                                if (
                                    config.AttributeEstimation.estimate_type
                                    == 'withKeyPoints'
                                    and crops_count_GT[i] != 0
                                ):
                                    maps_idx = save_keypoint_heatmap(
                                        image_name=image_name,
                                        crop_index=i,
                                        crop_image=crop_image,
                                        point_maps=sample_anns['points_annot'],
                                        predicted_maps=(
                                            all_predicted_detection_maps_toDraw
                                        ),
                                        maps_index=maps_idx,
                                        draw_maps=config.DrawProperties.DRAW_MAPS,
                                        maps_path=config.DrawProperties.maps_path,
                                    )

                    if verbose:
                        if not args.have_GT and len(estimation_outputs) >0 and is_roots_2:
                            for i in range(len(TRL_pred)):
                                printf(
                                    "color_pred: %.3f | TRL_pred: %.3f | dia_pred: %.3f \n",
                                    color_pred[i], TRL_pred[i], dia_pred[i]
                                )
                            print()

        summary_metrics = aggregate_post_loop_metrics(
            state,
            attributes=is_roots_2,
        )
        model.train()
        return finalize_evaluation(
            state,
            summary_metrics,
            attributes=is_roots_2,
            have_gt=args.have_GT,
            predict_empty_image=getattr(args, "predict_empty_image", False),
            verbose=verbose,
            print_to_files=print_to_files and args is not None,
            detection_metrics=detection_metrics,
            evaluate_points=evaluate_points,
            files_path=config.General.files_path,
        )









