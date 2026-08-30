"""Per-object detection and attribute-estimation evaluation utilities.

The public :func:`eval` entry point is retained for compatibility with the
training and inference pipelines.  Small private helpers keep matching rules
independently testable while the legacy evaluation routine is decomposed.
"""

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from thop import profile, clever_format

from legonet import config
from legonet.eval.evaluation_finalization import finalize_evaluation
from legonet.eval.image_context import prepare_image_context
from legonet.eval.model_input import build_per_object_model_input
from legonet.eval.prediction_summary import record_per_image_predictions
from legonet.eval.no_prediction_bookkeeping import record_no_predictions
from legonet.eval.detection_inputs import prepare_detection_evaluation_inputs
from legonet.eval.ground_truth_preparation import (
    prepare_image_ground_truth,
    split_boxes_by_annotations as _prepare_gt_boxes_for_attribute_eval,
)
from legonet.eval.evaluation_state import initiate_global_dicts
from legonet.eval.detection_bookkeeping import record_detection_bookkeeping
from legonet.eval.KP_detection_eval import (
    points_detection_t_p,
    process_keypoint_map_for_evaluation,
)
from legonet.eval.matching import (
    assign_detection_to_gt as _assign_detection_to_gt,
    choose_boxes_by_IoUandPrc,
    match_detections_to_gt as _match_detections_to_gt,
)
from legonet.eval.crop_predictions import prepare_crop_predictions
from legonet.eval.metric_aggregation import (
    PostLoopMetrics,
    aggregate_post_loop_metrics,
    compute_positive_crop_count_metrics as _compute_positive_crop_count_metrics,
)
from legonet.eval.visualization import (
    save_detection_overview,
    save_keypoint_heatmap,
    save_object_visualizations,
)
from legonet.my_dataloader import UnNormalizer
from legonet.progress import print_image_progress
from legonet.utils import printf



unnormalize = UnNormalizer()


def _include_crop_in_point_evaluation(
    ground_truth_map: torch.Tensor,
    *,
    evaluates_attributes: bool,
) -> bool:
    """Exclude empty GT crops from grape counting point evaluation."""
    return evaluates_attributes or torch.sum(ground_truth_map).item() > 0


def _evaluate_crop_keypoints(
    raw_map: torch.Tensor,
    processed_map: torch.Tensor,
    ground_truth_map: torch.Tensor,
    *,
    evaluates_attributes: bool,
) -> tuple[list[float], list[float]]:
    """Evaluate raw final-ReLU candidates for counting and root attributes."""
    return points_detection_t_p(
        raw_map,
        ground_truth_map,
        candidate_threshold=0.0,
    )


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
    matched_bbox_id=None,
):
    """Project the crop's applicable points onto a binary center map.

    Grape counting uses every geometrically contained point. Root attribute
    evaluation additionally restricts points to the matched object's bbox ID,
    which is the same target population used to supervise the root crop model.
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
        if (
            matched_bbox_id is not None
            and float(point.get("bbox_id", -1)) != float(matched_bbox_id)
        ):
            continue
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



def eval(
    dataset: Any,
    dataloader: Any,
    sampler: Any,
    model: Any,
    verbose: bool = True,
    to_draw: bool = True,
    draw_path: str | Path = "",
    print_to_files: bool = False,
    args: Any | None = None,
    do_profile: bool = False,
    detection_metrics: Sequence[Any] | None = None,
    evaluate_points: bool = False,
    return_metrics: bool = False,
) -> list[float] | PostLoopMetrics:
    """Evaluate per-object counting or attributes on a prepared dataset.

    The return value preserves the historical list consumed by training. Full
    metrics are reported through the configured text and structured artifacts.
    """
    if args is None:
        raise ValueError("Per-object evaluation requires configured run arguments.")

    evaluates_attributes = config.Detect_and_Estimate.type in {
        "per_object_attributes",
        "per_object_attributes_multibranch",
    }

    model.eval()

    draw_detection_overview = getattr(
        args, "draw_detection_overview", True
    )
    draw_gt_only = getattr(args, "draw_gt_only", False)
    draw_individual_objects = getattr(
        args, "draw_individual_object_visualizations", True
    )

    if to_draw:
        visualization_root = Path(draw_path)
        crops_path = visualization_root / "Predicted crops"
        crops_path.mkdir(parents=True, exist_ok=True)

        gt_path = visualization_root / "GT only"
        if draw_gt_only:
            gt_path.mkdir(parents=True, exist_ok=True)

        predicted_boxes_path = visualization_root / "Predicted boxes on image"
        if draw_detection_overview or draw_individual_objects:
            predicted_boxes_path.mkdir(parents=True, exist_ok=True)

    # gather all annotations, per image, per label
    if args.have_GT:
        all_box_annotations, all_count_annotations = _get_count_and_box_annotations(dataset)

    with torch.no_grad():

        state = initiate_global_dicts(initiate=True)
        print()
        for iter_num, data in enumerate(dataloader):

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
                attributes=evaluates_attributes,
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
                attributes=evaluates_attributes,
            )

            ###################################################################################################################################################
            # detection_outputs - outputs of the detection part (based on module where), after filtering by nms and min score
            # estimation_outputs - prediction of counting per box from detection_outputs
            # sample_anns - has the crop in its 'img' key. In training, it has also 'points_annot' key that holds the gt annotations per crop - relevant
            # for evaluation during training.
            ###################################################################################################################################################

            # bbox_pred - all predicted boxes - with or without points in it - rescaled to the orig img size
            obj_scores, bbox_pred = _get_detections(detection_outputs, scale) #bbox_pred coordinates for original image size

            if bbox_pred is None or len(bbox_pred)==0:
                if verbose:
                    printf("Image has no predicted boxes...\n")
                    print()

                record_no_predictions(
                    state,
                    image_name,
                    image_gt_average=im_gt_avg,
                    image_trl_sum=TRL_im_gt_sum,
                    image_diameter_average=dia_im_gt_avg,
                    attributes=evaluates_attributes,
                    network_type=config.Detect_and_Estimate.type,
                )

                continue

            detection_inputs = prepare_detection_evaluation_inputs(
                state,
                dataset,
                data,
                image_name,
                bbox_pred,
                obj_scores,
                scale,
                have_gt=args.have_GT,
                has_ground_truth_boxes=len(box_annotations_all) > 0,
            )
            point_anns = detection_inputs.point_annotations
            gt_boxes = detection_inputs.ground_truth_boxes

            if to_draw and (draw_detection_overview or draw_gt_only):
                save_detection_overview(
                    image_path=image_path,
                    image_name=image_name,
                    predicted_boxes=bbox_pred,
                    gt_boxes=gt_boxes,
                    point_annotations=point_anns,
                    scale=scale,
                    have_gt=args.have_GT,
                    draw_path=predicted_boxes_path,
                    gt_path=gt_path,
                    line_width=config.DrawProperties.LINE_WIDTH,
                    point_radius=config.DrawProperties.POINT_RADIUS,
                    draw_detection_overview=draw_detection_overview,
                    draw_gt_only=draw_gt_only,
                    draw_individual_objects=draw_individual_objects,
                )

            ############################################################################################################
            # Preparing the evaluation of the counting results per crop - doesn't depend on the detection performance -
            # we only need to find and evaluate the crops that include gt points - otherwise we'll compare
            # the predicted count to 0, probably since the points weren't annotated
            ############################################################################################################

            crop_predictions = prepare_crop_predictions(
                state=state,
                dataset=dataset,
                image_name=image_name,
                predicted_boxes=bbox_pred,
                original_crop_boxes=crops_orig_boxes,
                estimation_outputs=estimation_outputs,
                sample_annotations=sample_anns,
                scale=scale,
                evaluates_attributes=evaluates_attributes,
                have_ground_truth=args.have_GT,
                estimate_type=config.AttributeEstimation.estimate_type,
                crop_size=config.AttributeEstimation.crops_size,
                point_center_map_builder=_geometric_point_centers_map,
            )
            sample_anns = crop_predictions.sample_annotations
            adjusted_crops_orig_boxes = (
                crop_predictions.adjusted_original_boxes
            )
            crops_count_GT = crop_predictions.ground_truth_counts
            count_pred = crop_predictions.count_predictions
            length_pred = crop_predictions.length_predictions
            dia_pred = crop_predictions.diameter_predictions
            color_pred = crop_predictions.color_predictions
            all_predicted_detection_maps = (
                crop_predictions.predicted_detection_maps
            )
            all_predicted_detection_maps_toDraw = (
                crop_predictions.predicted_maps_to_draw
            )
            all_crops_GT_detections_maps = (
                crop_predictions.ground_truth_detection_maps
            )

            if np.sum(crops_count_GT)==0 and len(bbox_pred)>0: #sample_anns is None
                if verbose and args.have_GT:
                    printf("No gt points in any predicted crop...\n")
                    print()

            ############################################################################################################
            # Preparing the evaluation in comparison to the gt points' count of the relevant gt object -
            # the object with iou>thresh of the predicted box with the gt box
            ############################################################################################################

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
                attributes=evaluates_attributes,
                counting=(
                    config.Detect_and_Estimate.type == "per_object_counting"
                ),
                iou_threshold=config.Detection.iou_threshold,
            )
            orig_count_GT = bookkeeping.count_ground_truth
            orig_length_GT = bookkeeping.length_ground_truth
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
                                evaluation_maps = process_keypoint_map_for_evaluation(
                                    model, all_predicted_detection_maps
                                )
                                for b in range(evaluation_maps.shape[0]):
                                    if not _include_crop_in_point_evaluation(
                                        all_crops_GT_detections_maps[b],
                                        evaluates_attributes=evaluates_attributes,
                                    ):
                                        continue
                                    t, p = _evaluate_crop_keypoints(
                                        all_predicted_detection_maps[b, :, :],
                                        evaluation_maps[b, :, :],
                                        all_crops_GT_detections_maps[b, :, :],
                                        evaluates_attributes=evaluates_attributes,
                                    )
                                    state['T']=state['T']+ t
                                    state['P']=state['P']+ p
                                    if getattr(
                                        args,
                                        "compare_keypoint_protocols",
                                        False,
                                    ):
                                        comparison = state[
                                            "keypoint_protocol_comparison"
                                        ]
                                        for protocol_name in (
                                            "processed_threshold_0_02",
                                            "processed_local_maxima",
                                            "raw_nonzero",
                                        ):
                                            comparison.setdefault(
                                                protocol_name, {"T": [], "P": []}
                                            )
                                        processed_t, processed_p = points_detection_t_p(
                                            evaluation_maps[b, :, :],
                                            all_crops_GT_detections_maps[b, :, :],
                                        )
                                        local_t, local_p = points_detection_t_p(
                                            evaluation_maps[b, :, :],
                                            all_crops_GT_detections_maps[b, :, :],
                                            candidate_threshold=None,
                                            local_maxima_only=True,
                                        )
                                        raw_t, raw_p = points_detection_t_p(
                                            all_predicted_detection_maps[b, :, :],
                                            all_crops_GT_detections_maps[b, :, :],
                                            candidate_threshold=0.0,
                                        )
                                        comparison["processed_threshold_0_02"]["T"].extend(processed_t)
                                        comparison["processed_threshold_0_02"]["P"].extend(processed_p)
                                        comparison["processed_local_maxima"]["T"].extend(local_t)
                                        comparison["processed_local_maxima"]["P"].extend(local_p)
                                        comparison["raw_nonzero"]["T"].extend(raw_t)
                                        comparison["raw_nonzero"]["P"].extend(raw_p)

                    if not evaluates_attributes:
                        state['all_crops_GT_counts'].append(crops_count_GT)  # based on the number of points in the crop
                        state['all_predicted_counts'].append(np.round(count_pred))
                        state['all_orig_GT_counts'].append(orig_count_GT) # count in the corresponding annotated box, it is -1 if the crop is false positive

                    else:
                        state['all_predicted_lengths'].append(length_pred)
                        state['all_orig_GT_lengths'].append(orig_length_GT)

                        state['all_predicted_dia'].append(dia_pred)
                        state['all_orig_GT_dia'].append(orig_dia_GT)

                        state['all_predicted_color'].append(np.round(color_pred))
                        state['all_orig_GT_color'].append(orig_color_GT)

                    maps_idx=0
                    for i in range(len(crops_count_GT)):
                        if not evaluates_attributes:
                            if len(orig_count_GT) > 0:
                                if orig_count_GT[i] != -1:
                                    state['orig_abs_diff'].append(abs(orig_count_GT[i] - np.round(count_pred[i]))) #count_pred[i]))
                                    if orig_count_GT[i] > 0:
                                        state['orig_rel_error'].append(abs(orig_count_GT[i] - np.round(count_pred[i])) / orig_count_GT[i])

                        elif crops_count_GT[i] > 0:
                            if orig_color_GT[i] != -1:
                                state['orig_abs_diff_length'].append(abs(orig_length_GT[i] - length_pred[i]))
                                state['orig_rel_error_length'].append(abs(orig_length_GT[i] - length_pred[i]) / orig_length_GT[i])

                                state['orig_abs_diff_dia'].append(abs(orig_dia_GT[i] - dia_pred[i]))
                                state['orig_rel_error_dia'].append(abs(orig_dia_GT[i] - dia_pred[i]) / orig_dia_GT[i])

                                state['orig_abs_diff_color'].append(abs(orig_color_GT[i] - color_pred[i]))

                        if verbose:
                            if (not evaluates_attributes and len(orig_count_GT) > 0) \
                                    or (evaluates_attributes and len(orig_color_GT) > 0):

                                    if not evaluates_attributes:
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
                                                "orig_length_GT: %.3f | length_pred: %.3f | length_orig_abs_diff: %.3f | length_orig_rel_error:  %.3f\n",
                                                orig_length_GT[i], length_pred[i] , state['orig_abs_diff_length'][-1], state['orig_rel_error_length'][-1]
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

                    if (
                        to_draw
                        and sample_anns is not None
                    ):
                        for i in range(len(adjusted_crops_orig_boxes)):
                            bbox_crop = sample_anns['img'][i].clone()
                            gt_box_id = -1
                            if args.have_GT:
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
                                roots_attributes=evaluates_attributes,
                                image_points=point_anns,
                                crop_points=(
                                    relevant_points_anns[i]
                                    if args.have_GT
                                    and i < len(relevant_points_anns)
                                    else []
                                ),
                                has_positive_target=(
                                    args.have_GT
                                    and i < len(crops_count_GT)
                                    and crops_count_GT[i] != 0
                                ),
                                predicted_boxes_path=predicted_boxes_path,
                                crops_path=crops_path,
                                line_width=config.DrawProperties.LINE_WIDTH,
                                unnormalize=unnormalize,
                                save_predicted_box_overlay=(
                                    draw_individual_objects
                                ),
                            )

                            if (
                                config.AttributeEstimation.estimate_type
                                == 'withKeyPoints'
                                and (
                                    not args.have_GT
                                    or (
                                        i < len(crops_count_GT)
                                        and crops_count_GT[i] != 0
                                    )
                                )
                            ):
                                predicted_heatmap_value = (
                                    length_pred[i]
                                    if evaluates_attributes
                                    else count_pred[i]
                                )
                                heatmap_gt_values = (
                                    orig_length_GT
                                    if evaluates_attributes
                                    else orig_count_GT
                                )
                                ground_truth_heatmap_value = (
                                    heatmap_gt_values[i]
                                    if args.have_GT
                                    and i < len(heatmap_gt_values)
                                    and heatmap_gt_values[i] != -1
                                    else None
                                )
                                maps_idx = save_keypoint_heatmap(
                                    image_name=image_name,
                                    crop_index=i,
                                    crop_image=crop_image,
                                    point_maps=(
                                        sample_anns['points_annot']
                                        if args.have_GT
                                        else None
                                    ),
                                    predicted_maps=(
                                        all_predicted_detection_maps_toDraw
                                    ),
                                    maps_index=maps_idx,
                                    draw_maps=config.DrawProperties.DRAW_MAPS,
                                    maps_path=config.DrawProperties.maps_path,
                                    predicted_value=predicted_heatmap_value,
                                    ground_truth_value=ground_truth_heatmap_value,
                                    attribute_name=(
                                        "length" if evaluates_attributes else "count"
                                    ),
                                    attribute_unit=(
                                        "mm" if evaluates_attributes else ""
                                    ),
                                )

                    if verbose:
                        if not args.have_GT and len(estimation_outputs) >0 and evaluates_attributes:
                            for i in range(len(length_pred)):
                                printf(
                                    "color_pred: %.3f | length_pred: %.3f | dia_pred: %.3f \n",
                                    color_pred[i], length_pred[i], dia_pred[i]
                                )
                            print()

        summary_metrics = aggregate_post_loop_metrics(
            state,
            attributes=evaluates_attributes,
        )
        model.train()
        legacy_result = finalize_evaluation(
            state,
            summary_metrics,
            attributes=evaluates_attributes,
            have_gt=args.have_GT,
            predict_empty_image=getattr(args, "predict_empty_image", False),
            verbose=verbose,
            print_to_files=print_to_files,
            detection_metrics=detection_metrics,
            evaluate_points=evaluate_points,
            files_path=config.General.files_path,
            text_results_path=getattr(args, "txt_results", None),
        )
        return summary_metrics if return_metrics else legacy_result
