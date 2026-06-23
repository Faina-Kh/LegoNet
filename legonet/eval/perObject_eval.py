import torch
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import PIL
from PIL import Image
from torchvision import transforms
import config
import csv
from thop import profile, clever_format

from legonet.utils import printf
from legonet.eval.attribute_estimation_eval import SumOfAbsDifferences
from legonet.eval.detection_eval import compute_overlap, _compute_ap, plot_PR_curve
from legonet.eval.KP_detection_eval import points_detection_t_p, calc_points_recall_precision_ap, visualize_KeyPointsHeatmaps
from legonet.my_dataloader import UnNormalizer



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

    for i in range(len(generator)):
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


        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_box_annotations,all_count_annotations


def find_points_in_bbox(img, point_anns, bbox_pred, scale):

    points=[]

    # for drawing
    im = img.cpu().clone().detach()
    im = np.array(255 *unnormalize(im))
    im[im < 0] = 0
    im[im > 255] = 255
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB)

    for p in point_anns:
        p['x'] = p['x'] * scale
        p['y'] = p['y'] * scale

        cv2.circle(im, (int(p['x']), int(p['y'])), radius=3, color=(0, 0, 255), thickness=2)


    for b in range(bbox_pred.shape[0]):
        p_x = []
        p_y = []
        box_x1, box_y1, box_x2, box_y2 = bbox_pred[b]
        cv2.rectangle(im, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), color=(0, 0, 255), thickness=2)

        for p in point_anns:
            if p['x']<=box_x2 and p['x']>=box_x1 and p['y']<=box_y2 and p['y']>=box_y1:
                p_x.append(p['x'])
                p_y.append(p['y'])

        points.append({'x':p_x, 'y':p_y})

    # cv2.imshow('img', im)
    # cv2.waitKey(0)

    return points

def visualize_pointMaps(count_outputs, count_sample, image_name, imgToVis, maps_path):

    # Draw GT activations:
    img_copy = imgToVis.copy()
    background = img_copy.convert("RGBA")
    BG_w, BG_h = background.size

    background_2=img_copy.convert('L')   #convert image to monochrome
    background_2.save(maps_path + '/' + image_name + '_background.png')
    background_2 = Image.open(maps_path + '/' + image_name + '_background.png')
    background_2=background_2.convert("RGBA")

    if len(count_sample)>0:
        anno = count_sample.copy() #.cpu().numpy().copy()

        plt.imsave(maps_path + '/' + image_name + '_anno.png', anno)
        gt_anns = Image.open(maps_path + '/' + image_name + '_anno.png')
        gt_anns = gt_anns.resize((BG_w, BG_h), Image.Resampling.LANCZOS) #Image.ANTIALIAS)
        gt_anns.save(maps_path + '/' + image_name + '_anno.png')

        alphaBlended = Image.blend(gt_anns, background_2, 0.6)
        alphaBlended.save(maps_path + '/' + image_name + '_Blended_GT.png')

        os.remove(maps_path + '/' + image_name + '_anno.png')

    # Relu map #######################################################################################################

    if len(count_outputs) >0:
        plt.imsave(maps_path + '/' + image_name + '_Relu.png', count_outputs.cpu())
        relu_anns = Image.open(maps_path + '/' + image_name + '_Relu.png')

        relu_anns = relu_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
        relu_anns.save(maps_path + '/' + image_name + '_Relu.png')

        alphaBlended_relu = Image.blend(relu_anns, background_2.convert('RGBA'), 0.6)
        alphaBlended_relu.save(maps_path + '/' + image_name + '_Blended_Relu.png')

        os.remove(maps_path + '/' + image_name + '_Relu.png')

    os.remove(maps_path + '/' + image_name + '_background.png')

def nmcs(predicted_boxes, relevant_points):

    non_supressed_indices = [True for i in range(predicted_boxes.shape[0])]

    for i in range(predicted_boxes.shape[0]):
        current_points = relevant_points[i]

        if len(current_points["x"]) == 0:  # no relevant points
            continue

        for j in range(predicted_boxes.shape[0]):

            candidate_points = relevant_points[j]

            if i == j or len(candidate_points["x"]) == 0 or non_supressed_indices[j] == False:
                continue
            else:
                common_points_count = 0
                for m in range(len(candidate_points["x"])):
                    x1, y1 = candidate_points["x"][m], candidate_points["y"][m],
                    for n in range(len(current_points["x"])):
                        x2, y2 = current_points["x"][n], current_points["y"][n],
                        if x1 == x2 and y1 == y2:
                            common_points_count += 1
                if common_points_count == len(candidate_points["x"]):
                    non_supressed_indices[j] = False

    return non_supressed_indices

def choose_boxes_by_IoUandPrc(detections, annotations, d_scores):
    #annotations = annotations[0,:,:4]
    annotations_boxes = annotations[0,:,:4]
    iou_threshold = config.Detection.iou_threshold
    precision_thresh = config.Detect_and_Estimate.precision_thresh

    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    num_annotations = annotations.shape[0]
    detected_annotations = []

    indices = np.argsort(-d_scores.cpu())
    detections = detections[indices, :]

    detection_assignments = []

    for d in detections:
        if annotations_boxes.shape[0] == 0:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)
            continue

        overlaps = compute_overlap(np.expand_dims(d.cpu(), axis=0), annotations_boxes.cpu().numpy())
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]

        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
            false_positives = np.append(false_positives, 0)
            true_positives = np.append(true_positives, 1)
            detected_annotations.append(assigned_annotation)

            #detection_assignments.append(torch.cat((d.unsqueeze(0), torch.tensor([float(assigned_annotation)]).unsqueeze(0).to(config.General.device)), dim=-1))
            detection_assignments.append(torch.cat(
                (d.unsqueeze(0), torch.tensor([float(annotations[0][assigned_annotation[0]][5])]).unsqueeze(0).to(config.General.device)),
                dim=-1))

        else:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)

            detection_assignments.append(torch.cat((d.unsqueeze(0),torch.tensor([-1.0]).unsqueeze(0).to(config.General.device)), dim=-1))


    # changes for roots
    return detection_assignments



    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        mAP = 0
    else:

        # sort by score
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        mAP = _compute_ap(recall, precision)

        # sort by precision
        precision_idx= np.argsort(-precision)
        precision2=precision[precision_idx]
        recall2=recall[precision_idx]

        relevant_idx=np.where(precision2>precision_thresh)[0]
        if len(relevant_idx) >0:
            relevant_idx=relevant_idx[-1]

            pr= precision2[relevant_idx]
            rc=recall2[relevant_idx]
            #print('mAP = {:0.3f}, th={}, pr={}, rc={}\n'.format(mAP, precision_thresh, pr, rc))
            detections=detections[:(relevant_idx+1)]

            return detections

        else:
            #print('No relevant detections...\n')
            return []

def view_points_on_img(img, point_anns):

    for p in point_anns:
        cv2.circle(img, (int(p['x']), int(p['y'])), radius=3, color=(0, 0, 255), thickness=2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


##########################################################################################
def image_output_shape(image_shape, pyramid_level=3):
    return (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)

def images_ratios(image_shape, output_shape):
    return output_shape / np.array(image_shape[:2])

def create_gausian_mask(center_point, nCols, nRows, q=99, radius=(5, 5)):
    '''
    create_gausian_mask creates a gaussian mask to be used as GT annotations for the detection-based counter
    :param center_point:
    :param nCols:
    :param nRows:
    :param q:
    :param s:
    :param radius:
    :return:
    '''
    s = 3
    # if (s >= radius[0]):
    #     s = 1
    x = np.tile(range(nCols), (nRows, 1))
    y = np.tile(np.reshape(range(nRows), (nRows, 1)), (1, nCols))

    x2 = (((x - np.round(center_point[0])) * s) / radius[0]) ** 2
    y2 = (((y - np.round(center_point[1])) * s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return p

def compute_keypoints_targets_multi_maps(image_shape, annotations_points_centers_a, radius=(5, 5), pyramid_level=3):
    # resize transformed-image and annotations
    import copy
    annotations_points_centers = copy.deepcopy(annotations_points_centers_a)

    # here we should resize image too and then check it with the annotations
    output_shape = image_output_shape(image_shape[2:], pyramid_level=pyramid_level)

    image_ratio = images_ratios(image_shape[2:], output_shape)

    if len(annotations_points_centers) == 0:
        return [np.zeros(output_shape)]

    per_img_anns = []
    img_num = len(annotations_points_centers)
    for i in range(img_num):
        current_points_a = annotations_points_centers[i]  # [N, [x, y, points class, bbox_id]]
        current_points = np.array([current_points_a['x'], current_points_a['y']])
        current_points[0] = current_points[0] * image_ratio[0]
        current_points[1] = current_points[1] * image_ratio[1]
        annotations = np.zeros(output_shape)
        for j in range(current_points[0].shape[0]):
            gaussian_map = create_gausian_mask((current_points[0,j],current_points[1,j]), output_shape[1], output_shape[0],
                                                    radius=radius)
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)

        if np.isnan(annotations).any():
            raise ("nan was found")

        per_img_anns.append(annotations)

    return per_img_anns

def objects_recall_precision(all_annotations, all_detections):

    average_precisions = {}

    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0

    for i in range(len(all_annotations)):
        detections = all_detections[i]
        annotations = all_annotations[i]

        if len(annotations) >0:
            if len(detections)>0:

                if len(annotations.shape) == 1:
                    annotations = np.expand_dims(annotations, axis=0)

                num_annotations += annotations.shape[0]
                detected_annotations = []

                scores_temp = np.zeros((0,))

                for d in detections:
                  scores_temp = np.append(scores_temp, d[4])

                indices = np.argsort(-scores_temp)

                detections = detections[indices,:]

                for d in detections:
                    scores = np.append(scores, d[4])

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= config.Detection.iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            else:
                scores = np.append(scores, 0)
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 0)

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    mAP = _compute_ap(recall, precision)

    plot_PR_curve(recall, precision, mAP, save_path= config.General.files_path,  plots_name = "objects_PR_curve.png") #"D:\\Faina\\Parts count papers\\paper 2\\docs\\grapes")#draw_path)
    # plt.plot(recall, precision)
    # plt.title('mAP = {:.2f}'.format(np.mean(mAP)))
    # plt.grid(True)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.savefig(os.path.join(draw_path,"objects_PR_curve.png"))

    return mAP, precision, recall

def initiate_global_dicts(state=None, image_name='', initiate=False):
    """
        Initialize or extend the evaluation state dictionary.

        Args:
            state (dict | None):
                Existing state dictionary. If None and initiate=True, a new state is created.
            image_name (str):
                Image name to initialize per-image fields for when initiate=False.
            initiate (bool):
                If True, create and return a fresh initialized state.
                If False, add per-image entries to the existing state.

        Returns:
            dict: Updated state dictionary.
        """

    is_roots = (config.Detect_and_Estimate.type == "both")
    is_roots_2 = (config.Detect_and_Estimate.type == "both_for_roots_2" or config.Detect_and_Estimate.type =="both_Back2bFind2b")

    if initiate:
        state = {
            # counting
            'all_predicted_counts': [],
            'T': [],
            'P': [],
            'all_crops_GT_counts': [],
            'crops_abs_diff': [],
            'crops_rel_error': [],

            'all_orig_GT_counts': [],
            'orig_abs_diff': [],
            'orig_rel_error': [],

            'all_data_gt_count': [],
            'gt_objects_withGTpoints': 0,
            'found_orig_objects': 0,
            'FP': 0,
            'predicted_counts_any_crop': [],
            'matched_without_gt_points': 0,
            'crops_without_gt_points': 0,
            'detections_data_any_crop': {},

            'not_found_gt': {},
            'no_predictions': {},

            'per_im_gt_avg': [],
            'per_im_gt_avg_dict': {},
            'per_im_pred_avg': [],
            'per_im_pred_dict': {},

            'num_of_gt_boxes': 0,
            'all_detections': [],
            'all_annotations': []
        }

        if is_roots_2:
            state.update({
                'all_predicted_TRL': [],
                'all_predicted_dia': [],
                'all_predicted_color': [],

                'all_crops_GT_TRL': [],
                'crops_abs_diff_TRL': [],
                'crops_rel_error_TRL': [],

                'all_crops_GT_dia': [],
                'crops_abs_diff_dia': [],
                'crops_rel_error_dia': [],

                'all_orig_GT_TRL': [],
                'orig_abs_diff_TRL': [],
                'orig_rel_error_TRL': [],

                'all_orig_GT_dia': [],
                'orig_abs_diff_dia': [],
                'orig_rel_error_dia': [],

                'all_orig_GT_color': [],
                'orig_abs_diff_color': [],

                'all_data_gt_TRL': [],
                'predicted_TRL_any_crop': [],

                'all_data_gt_dia': [],
                'predicted_dia_any_crop': [],

                'all_data_gt_color': [],
                'predicted_color_any_crop': [],

                'TRL_per_im_gt_sum': [],
                'TRL_per_im_gt_sum_dict': {},
                'TRL_per_im_pred_sum': [],
                'TRL_per_im_pred_dict': {},

                'dia_per_im_gt_avg': [],
                'dia_per_im_gt_avg_dict': {},
                'dia_per_im_pred_avg': [],
                'dia_per_im_pred_dict': {},
            })

        return state

    if state is None:
        raise ValueError("state must be provided when initiate=False")

    if image_name == '':
        raise ValueError("image_name must be provided when initiate=False")

    state['detections_data_any_crop'][image_name] = {
        'gt_count': [],
        'label': [],
        'score': [],
        'gt_box_id': [],
        'max_overlap': []
    }

    state['not_found_gt'][image_name] = {
        #'pred': [],
        'gt_count': [],
        'label': [],
        'score': [],
        'max_overlap': []
    }

    if is_roots:
        state['detections_data_any_crop'][image_name].update({
            'pred': []})
        state['not_found_gt'][image_name].update({
            'pred': []})

    if is_roots_2:
        state['detections_data_any_crop'][image_name].update({
            'color_pred': [],
            'color_gt': [],
            'TRL_pred': [],
            'TRL_gt': [],
            'dia_pred': [],
            'dia_gt': []
        })

        state['not_found_gt'][image_name].update({
            'color_pred': [],
            'color_gt': [],
            'TRL_pred': [],
            'TRL_gt': [],
            'dia_pred': [],
            'dia_gt': []
        })

    return state

    # if initiate:
    #     # for counting
    #     all_predicted_counts = []
    #     T, P = [], []
    #     all_crops_GT_counts = []
    #     crops_abs_diff = []
    #     crops_rel_error = []
    #
    #     all_orig_GT_counts = []
    #     orig_abs_diff = []
    #     orig_rel_error = []
    #
    #     all_data_gt_count = []
    #     gt_objects_withGTpoints = 0
    #     found_orig_objects = 0
    #     FP = 0
    #     predicted_counts_any_crop = []
    #     matched_without_gt_points = 0
    #     crops_without_gt_points = 0
    #     detections_data_any_crop = {}
    #
    #     not_found_gt = {}
    #     no_predictions = {}
    #
    #     per_im_gt_avg = []
    #     per_im_gt_avg_dict = {}
    #     per_im_pred_avg = []
    #     per_im_pred_dict = {}
    #
    #     num_of_gt_boxes = 0
    #     all_detections = []
    #     all_annotations = []
    #
    #     if config.Detect_and_Estimate.type == "both_for_roots_2":
    #         all_predicted_TRL = []
    #         all_predicted_dia = []
    #         all_predicted_color = []
    #
    #         all_crops_GT_TRL = []
    #         crops_abs_diff_TRL = []
    #         crops_rel_error_TRL = []
    #
    #         all_crops_GT_dia = []
    #         crops_abs_diff_dia = []
    #         crops_rel_error_dia = []
    #
    #         all_orig_GT_TRL = []
    #         orig_abs_diff_TRL = []
    #         orig_rel_error_TRL = []
    #
    #         all_orig_GT_dia = []
    #         orig_abs_diff_dia = []
    #         orig_rel_error_dia = []
    #
    #         all_orig_GT_color = []
    #         orig_abs_diff_color = []
    #
    #         all_data_gt_TRL = []
    #         predicted_TRL_any_crop = []
    #
    #         all_data_gt_dia = []
    #         predicted_dia_any_crop = []
    #
    #         all_data_gt_color = []
    #         predicted_color_any_crop = []
    #
    #         TRL_per_im_gt_sum = []
    #         TRL_per_im_gt_sum_dict = {}
    #         TRL_per_im_pred_sum = []
    #         TRL_per_im_pred_dict = {}
    #
    #         dia_per_im_gt_avg = []
    #         dia_per_im_gt_avg_dict = {}
    #         dia_per_im_pred_avg = []
    #         dia_per_im_pred_dict = {}
    #
    # else:
    #     detections_data_any_crop[image_name] = {
    #         'pred': [],
    #         'gt_count': [],
    #         'label': [],
    #         'score': [],
    #         'gt_box_id': [],
    #         'max_overlap': []
    #     }
    #
    #     not_found_gt[image_name] = {
    #         'pred': [],
    #         'gt_count': [],
    #         'label': [],
    #         'score': [],
    #         'max_overlap': []
    #     }
    #
    #     if config.Detect_and_Estimate.type == "both_for_roots_2":
    #         detections_data_any_crop[image_name].update({
    #             'color_pred': [],
    #             'color_gt': [],
    #             'TRL_pred': [],
    #             'TRL_gt': [],
    #             'dia_pred': [],
    #             'dia_gt': []
    #         })
    #
    #         not_found_gt[image_name].update({
    #             'color_pred': [],
    #             'color_gt': [],
    #             'TRL_pred': [],
    #             'TRL_gt': [],
    #             'dia_pred': [],
    #             'dia_gt': []
    #         })




def eval(dataset, dataloader, sampler, model, verbose=True, to_draw=True, draw_path= "",
         print_to_files=False, args = None, do_profile = False):
    is_roots_2 = (
                config.Detect_and_Estimate.type == "both_for_roots_2" or config.Detect_and_Estimate.type == "both_Back2bFind2b")

    model.eval()

    #if to_draw:
    #draw_path = os.path.join("D:\\Faina\\Parts count papers\\paper 2\\docs", "grapes", 'vis_4_6_21')  #config.General.experiment_path, 'vis')
    if to_draw: #draw_path!= "":
        # if not os.path.exists(draw_path):
        #     os.makedirs(draw_path)
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
                #crops_TRL_GT = []
                orig_TRL_GT = []
                TRL_pred = []
                #crops_dia_GT = []
                orig_dia_GT = []
                dia_pred = []
                color_pred = []
                orig_color_GT = []

            image = data['img'].clone().detach()
            scale= data['scale']

            group_idx=sampler.groups[iter_num]
            img_id = dataset.image_ids[group_idx[0]]
            image_name = dataset.img_info[img_id]['name']
            image_path = os.path.join(dataset.base_dir, image_name)

            # get image annotations - bboxes and counts
            if args.have_GT:
                box_annotations_temp = all_box_annotations[img_id][0]
                gt_counts_temp = all_count_annotations[img_id][0]
            else:
                gt_counts_temp = []

            if len(gt_counts_temp)>0 or is_roots_2:
                if len(gt_counts_temp)>0:
                    im_gt_avg = np.sum(gt_counts_temp[:, 0]) / gt_counts_temp.shape[0]
                    state['per_im_gt_avg'].append(im_gt_avg)
                    state['per_im_gt_avg_dict'][image_name] = im_gt_avg

                    if is_roots_2:
                        TRL_im_gt_sum = np.sum(gt_counts_temp[:, 3]) #/ gt_counts_temp.shape[0]
                        state['TRL_per_im_gt_sum'].append(TRL_im_gt_sum)
                        state['TRL_per_im_gt_sum_dict'][image_name] = TRL_im_gt_sum

                        dia_im_gt_avg = np.sum(gt_counts_temp[:, 4]) / gt_counts_temp.shape[0]
                        state['dia_per_im_gt_avg'].append(dia_im_gt_avg)
                        state['dia_per_im_gt_avg_dict'][image_name] = dia_im_gt_avg

                else:
                    im_gt_avg = 0
                    state['per_im_gt_avg'].append(im_gt_avg)
                    state['per_im_gt_avg_dict'][image_name] = im_gt_avg

                    if is_roots_2:
                        TRL_im_gt_sum = 0 # / gt_counts_temp.shape[0]
                        state['TRL_per_im_gt_sum'].append(TRL_im_gt_sum)
                        state['TRL_per_im_gt_sum_dict'][image_name] = TRL_im_gt_sum

                        dia_im_gt_avg = 0
                        state['dia_per_im_gt_avg'].append(dia_im_gt_avg)
                        state['dia_per_im_gt_avg_dict'][image_name] = dia_im_gt_avg

            box_annotations_withPoints = []
            box_annotations_all = []
            gt_counts=[]

            state = initiate_global_dicts(state, image_name)

            printf(
                "##############################################################################################\n")
            printf("image: %s\n", image_name)
            printf("##############################################################################################\n")

            if args.have_GT:

                for i in range(len(box_annotations_temp)): # box_annotations_temp - all gt boxes
                    b=box_annotations_temp[i]
                    id=box_annotations_temp[i][5]

                    box_annotations_all.append(torch.tensor(b).unsqueeze(dim=0))
                    state['num_of_gt_boxes']+=1

                    #filtering the annotations - prevents having gt boxes without gt points:
                    for g in gt_counts_temp: # gt_counts_temp -all gt points
                        if g[2] == id:
                            gt_counts.append(g)
                            box_annotations_withPoints.append(torch.tensor(b).unsqueeze(dim=0))
                            break

                if len(box_annotations_withPoints)==0 and len(box_annotations_all) > 0:
                    if verbose:
                        # printf(
                        #     "##############################################################################################\n")
                        # printf("image: %s\n", image_name)
                        # printf("##############################################################################################\n")
                        printf("Has gt boxes but no gt points in any gt box...\n")
                        printf("Skipping this image...\n")
                        print()

                    #if not config.General.predict_empty_image: #is_roots_2:
                        continue

                if len(box_annotations_all) == 0:
                    if verbose:
                        # printf(
                        #     "##############################################################################################\n")
                        # printf("image: %s\n", image_name)
                        # printf(
                        #     "##############################################################################################\n")
                        printf("No gt boxes ...\n")
                        print()

                    if not config.General.predict_empty_image: #is_roots_2:
                        continue

                #from list to tensor:
                if len(box_annotations_withPoints)>1:
                    box_annotations_withPoints = torch.cat(box_annotations_withPoints, dim=0)

                elif len(box_annotations_withPoints)==1:
                    box_annotations_withPoints=box_annotations_withPoints[0]

                if len(box_annotations_all)>1:
                    box_annotations_all = torch.cat(box_annotations_all, dim=0)
                else:
                    if len(box_annotations_all)>0:
                        box_annotations_all=box_annotations_all[0]

                ##############################################################
                # get stats - count gt boxes with gt points
                for c in gt_counts:
                    state['all_data_gt_count'].append(c[0])
                    state['gt_objects_withGTpoints'] += 1

                    if is_roots_2:
                        state['all_data_gt_TRL'].append(c[3])
                        state['all_data_gt_dia'].append(c[4])
                        state['all_data_gt_color'].append(c[0])


            # run the network

            #image.cuda().float().unsqueeze(dim=0))

            if not args.have_GT:
                input = [image.to(config.General.device).float(), None, torch.tensor(group_idx)] #, True]
                detection_outputs, estimation_outputs, sample_anns, relevant_points_anns, crops_orig_boxes = model(input)


            elif 'points_annot' in data.keys():  # true when evaluating during training
                input = [image.to(config.General.device).float(), [data['bbox_annot'], data['points_annot']],
                         torch.tensor(group_idx)]  # ,True]
                detection_outputs, estimation_outputs, sample_anns, relevant_points_anns, crops_orig_boxes = \
                    model(input)

            else:
                input = [image.to(config.General.device).float(), [data['bbox_annot'], None],
                         torch.tensor(group_idx)]  # , True] #image.cuda().float().unsqueeze(dim=0))
                detection_outputs, estimation_outputs, sample_anns, relevant_points_anns, crops_orig_boxes = \
                    model(input)



            if iter_num==0 and do_profile:
                print("Both_2 FLOPS:")

                # Use thop to profile the model
                input = input
                flops, params = profile(model, inputs=(input,))

                # Print the estimated FLOPS and parameters
                flops_str, params_str = clever_format([flops, params], "%.3f")
                print(f"FLOPS: {flops_str}")
                print(f"Params: {params_str}")

            # from torchsummary import summary
            # summary(model, params = [(3,832,1088), [data['bbox_annot'], data['points_annot']], torch.tensor(group_idx), True] )

            # get predictions stats
            current_sum = 0 #for count estimate
            if is_roots_2:
                current_TRL_sum = 0
                current_dia_sum = 0

            if estimation_outputs is not None:
                for c_out in estimation_outputs[0]:
                    if is_roots_2:
                        current_pred = np.round(c_out.cpu()[0].numpy())

                        current_TRL_pred = c_out.cpu()[1].numpy()
                        state['predicted_TRL_any_crop'].append(current_TRL_pred)
                        current_TRL_sum += current_TRL_pred

                        current_dia_pred = c_out.cpu()[2].numpy()
                        state['predicted_dia_any_crop'].append(current_dia_pred)
                        current_dia_sum += current_dia_pred

                        current_color_pred = np.round(c_out.cpu()[0].numpy())
                        state['predicted_color_any_crop'].append(current_color_pred)

                    else:
                        current_pred = np.round(c_out.cpu().item()) #c_out.cpu().numpy()[0]

                    state['predicted_counts_any_crop'].append(current_pred)
                    current_sum += current_pred


                state['per_im_pred_avg'].append(current_sum/estimation_outputs[0].shape[0])  # relevant only to counting
                state['per_im_pred_dict'][image_name] = state['per_im_pred_avg'][-1]

                if is_roots_2:
                    state['TRL_per_im_pred_sum'].append(current_TRL_sum) # / estimation_outputs[0].shape[0])
                    state['TRL_per_im_pred_dict'][image_name] = state['TRL_per_im_pred_sum'][-1]
                    state['dia_per_im_pred_avg'].append(current_dia_sum / estimation_outputs[0].shape[0])
                    state['dia_per_im_pred_dict'][image_name] = state['dia_per_im_pred_avg'][-1]


            ###################################################################################################################################################
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
                adjusted_crops_orig_boxes = []

                if verbose:
                    printf("Image has no predicted boxes...\n")
                    print()

                if not image_name in state['no_predictions'].keys():

                    state['no_predictions'][image_name]= {
                   'gt_count': [], 'label': [], 'score': [], 'max_overlap': []
                    } #'pred': [],

                    if config.Detect_and_Estimate.type == "both":
                        state['no_predictions'][image_name].update({'pred': []})

                    if is_roots_2:
                        state['no_predictions'][image_name].update({
                            'TRL_pred': [], 'TRL_gt': [], 'dia_pred': [], 'dia_gt': [], 'color_pred': [], 'color_gt': []
                        })

                if config.Detect_and_Estimate.type == "both":
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


                #orig_img.show(title=image_name)
                orig_img.save(os.path.join(draw_path, image_name))


            ############################################################################################################
            # Preparing the evaluation of the counting results per crop - doesn't depend on the detection performance -
            # we only need to find and evaluate the crops that include gt points - otherwise we'll compare
            # the predicted count to 0, probably since the points weren't annotated
            ############################################################################################################

            #check_None = 1
             # sample_anns: corrected gt - has gt annotations for the crops  - for eval in training
            #if sample_anns is not None:
            #    if 'points_annot' in sample_anns.keys():
            #        check_None = 0

                    #ToDo - fix this part - currently refers only to predicted crops with gt points in them - not correct
            #        aaa = 1

                    # evaluating during training
                    # crops_count_GT = np.array(sample_anns['points_annot'][0].cpu())
                    # all_crops_GT_detections_maps = sample_anns['points_annot'][5]
                    # all_predicted_detection_maps = estimation_outputs[6]
                    # for b in range(estimation_outputs[0].shape):
                    #     count_pred[image_name]['pred'].append(np.round(estimation_outputs[0][b].cpu().item())) # estimation_outputs[0][b].cpu().item())
                    #     detections_data_any_crop.append(count_pred[-1])

            #if check_None: #evaluation

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

                    # x1 = bbox_pred[b, 0]
                    # y1 = bbox_pred[b, 1]
                    # x2 = bbox_pred[b, 2]
                    # y2 = bbox_pred[b, 3]

                    #if config.Detect_and_Estimate.type == "both_for_roots_2":

                            #state['detections_data_any_crop'][image_name]['pred'].append(np.round(estimation_outputs[0][b][0].cpu().item())) # [0] - color, [1] - length, [2] - dia
                        if estimation_outputs is not None:
                            if is_roots_2:
                                state['detections_data_any_crop'][image_name]['color_pred'].append(estimation_outputs[0][b][0].cpu().item())
                                state['detections_data_any_crop'][image_name]['TRL_pred'].append(estimation_outputs[0][b][1].cpu().item())
                                state['detections_data_any_crop'][image_name]['dia_pred'].append(estimation_outputs[0][b][2].cpu().item())

                                color_pred.append(estimation_outputs[0][b][0].cpu().item())
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
                                    sample_anns['points_annot'][5][b])  # sample_anns['points_annot'][-1][5][0]

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
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints': # and config.DrawProperties.DRAW_MAPS:
                        # if not config.Detect_and_Estimate.type == "both_for_roots_2":
                        #     all_crops_GT_detections_maps = torch.tensor(all_crops_GT_detections_maps)
                        # elif args.have_GT:
                        if args.have_GT:
                            all_crops_GT_detections_maps = torch.cat([torch.unsqueeze(map, dim=0) for map in all_crops_GT_detections_maps], dim=0)

                        if len(all_predicted_detection_maps)>1:
                            all_predicted_detection_maps = torch.cat([torch.unsqueeze(map, dim=0) for map in all_predicted_detection_maps], dim=0)
                        else:
                            all_predicted_detection_maps = all_predicted_detection_maps[0].unsqueeze(dim=0)

            if np.sum(crops_count_GT)==0 and len(bbox_pred)>0: #sample_anns is None
                if verbose and args.have_GT:
                    # printf(
                    #     "##############################################################################################\n")
                    # printf("image: %s\n", image_name)
                    # printf(
                    #     "##############################################################################################\n")
                    printf("No gt points in any predicted crop...\n")
                    print()

                if not is_roots_2:
                    continue



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

            gt_counts_copy = gt_counts.copy()

            remove_found_ids = []

            if len(adjusted_crops_orig_boxes)>0:

                if len(box_annotations_withPoints)>0:

                    detected_annotations = []
                    scores_temp = np.zeros((0,))

                    for i in range(len(adjusted_crops_orig_boxes)):
                        scores_temp = np.append(scores_temp, adjusted_crops_orig_boxes[i][4])
                        adjusted_crops_orig_boxes[i] = torch.tensor(adjusted_crops_orig_boxes[i]).unsqueeze(dim=0)

                    adjusted_crops_orig_boxes = torch.cat(adjusted_crops_orig_boxes, dim=0)
                    indices = np.argsort(-scores_temp)
                    adjusted_crops_orig_boxes = adjusted_crops_orig_boxes[indices]


                    max_overlap_array = []

                    for d in adjusted_crops_orig_boxes:
                        overlaps = compute_overlap(np.expand_dims(d, axis=0), np.array(box_annotations_withPoints))
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap = overlaps[0, assigned_annotation]

                        state['detections_data_any_crop'][image_name]['score'].append(float(d[4]))
                        state['detections_data_any_crop'][image_name]['max_overlap'].append(max_overlap[0])

                        if max_overlap >= config.Detection.iou_threshold and assigned_annotation not in detected_annotations:
                            detected_annotations.append(assigned_annotation)
                            max_overlap_array.append(max_overlap)
                            state['found_orig_objects'] += 1

                            state['detections_data_any_crop'][image_name]['label'].append(1)

                            gt_box_id = box_annotations_withPoints[assigned_annotation[0]][5]

                            state['detections_data_any_crop'][image_name]['gt_box_id'].append(gt_box_id)

                            # find the gt count value of the assigned gt box
                            has_points=False
                            for g in range(len(gt_counts)):
                                if gt_counts[g][2] == gt_box_id:
                                    if not is_roots_2:
                                        gt_count = gt_counts[g][0]
                                        orig_count_GT.append(gt_count)
                                        state['detections_data_any_crop'][image_name]['gt_count'].append(gt_count)

                                    has_points=True

                                    if is_roots_2:
                                        gt_TRL = gt_counts[g][3]
                                        orig_TRL_GT.append(gt_TRL)
                                        gt_dia = gt_counts[g][4]
                                        orig_dia_GT.append(gt_dia)
                                        gt_color = gt_counts[g][0]
                                        orig_color_GT.append(gt_color)


                                    #collect the found gt indexes in gt_counts
                                    remove_found_ids.append(g)

                                    if is_roots_2:
                                        state['detections_data_any_crop'][image_name]['color_gt'].append(gt_color)
                                        state['detections_data_any_crop'][image_name]['TRL_gt'].append(gt_TRL)
                                        state['detections_data_any_crop'][image_name]['dia_gt'].append(gt_dia)

                                    break

                            # matched a gt box without gt points
                            if not has_points:

                                #state['matched_without_gt_points']+=1
                                if not is_roots_2:
                                    orig_count_GT.append(-1)
                                    state['detections_data_any_crop'][image_name]['gt_count'].append(0)

                                else:
                                    orig_TRL_GT.append(-1)
                                    state['detections_data_any_crop'][image_name]['TRL_gt'].append(0)
                                    orig_dia_GT.append(-1)
                                    state['detections_data_any_crop'][image_name]['dia_gt'].append(0)
                                    orig_color_GT.append(-1)
                                    state['detections_data_any_crop'][image_name]['color_gt'].append(-1)


                        else:
                            if not is_roots_2:
                                orig_count_GT.append(-1)
                            # max_overlap_array.append(-1)
                            max_overlap_array.append(max_overlap)
                            state['FP'] += 1
                            state['detections_data_any_crop'][image_name]['label'].append(0)
                            if not is_roots_2:
                                state['detections_data_any_crop'][image_name]['gt_count'].append(-1)

                            state['detections_data_any_crop'][image_name]['gt_box_id'].append(torch.tensor(-1, dtype=float))

                            if is_roots_2:
                                orig_TRL_GT.append(-1)
                                state['detections_data_any_crop'][image_name]['TRL_gt'].append(-1)
                                orig_dia_GT.append(-1)
                                state['detections_data_any_crop'][image_name]['dia_gt'].append(-1)
                                orig_color_GT.append(-1)
                                state['detections_data_any_crop'][image_name]['color_gt'].append(-1)

                else:

                    max_overlap_array=None

                    for i in range(len(adjusted_crops_orig_boxes)):

                        state['FP'] += 1

                        state['detections_data_any_crop'][image_name]['score'].append(-1)
                        state['detections_data_any_crop'][image_name]['max_overlap'].append(-1)

                        state['detections_data_any_crop'][image_name]['label'].append(0)
                        if not is_roots_2:
                            state['detections_data_any_crop'][image_name]['gt_count'].append(-1)

                        state['detections_data_any_crop'][image_name]['gt_box_id'].append(torch.tensor(-1, dtype=float))

                        if is_roots_2:
                            orig_TRL_GT.append(-1)
                            state['detections_data_any_crop'][image_name]['TRL_gt'].append(-1)
                            orig_dia_GT.append(-1)
                            state['detections_data_any_crop'][image_name]['dia_gt'].append(-1)

                            orig_color_GT.append(-1)
                            state['detections_data_any_crop'][image_name]['color_gt'].append(-1)


            # get the not found boxes (that have points)
            # remove those that were found
            all = np.arange(len(gt_counts_copy))
            if len(remove_found_ids)>0:
                keep_not_found = list(set(all) - set(remove_found_ids))
            else:
                keep_not_found=all

            if len(keep_not_found)>0:
                for i in keep_not_found:
                    if config.Detect_and_Estimate.type == "both":
                        state['not_found_gt'][image_name]['gt_count'].append(gt_counts_copy[i][0])
                        state['not_found_gt'][image_name]['pred'].append(-1)

                    state['not_found_gt'][image_name]['label'].append(1)
                    state['not_found_gt'][image_name]['score'].append(-1)
                    state['not_found_gt'][image_name]['max_overlap'].append(-1)


                    if is_roots_2:
                        state['not_found_gt'][image_name]['TRL_gt'].append(gt_counts_copy[i][3])
                        state['not_found_gt'][image_name]['TRL_pred'].append(-1)
                        state['not_found_gt'][image_name]['dia_gt'].append(gt_counts_copy[i][4])
                        state['not_found_gt'][image_name]['dia_pred'].append(-1)

                        state['not_found_gt'][image_name]['color_gt'].append(gt_counts_copy[i][0])
                        state['not_found_gt'][image_name]['color_pred'].append(-1)


            # add info of not found gt boxes without gt points?
            # gt_boxes_without_points_num=len(box_annotations_all)-len(box_annotations_withPoints)
            # if gt_boxes_without_points_num>0:
            #     for i in range(gt_boxes_without_points_num):
            #         state['not_found_gt'][image_name]['gt_count'].append(0)
            #         state['not_found_gt'][image_name]['pred'].append(-1)
            #         state['not_found_gt'][image_name]['label'].append(1)
            #         state['not_found_gt'][image_name]['score'].append(-1)


            ############################################################################################################
            # Evaluate the results of the given image
            ############################################################################################################
            if estimation_outputs is not None:
                if len(estimation_outputs):
                    if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                        if config.AttributeEstimation.calc_det_performance:
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



                    # if verbose:
                    #     if len(crops_count_GT)>0 or (not args.have_GT and len(estimation_outputs) >0) :
                    #         printf("#######################################################################################\n")
                    #         printf("image: %s\n", image_name)
                    #         printf("#######################################################################################\n")

                    maps_idx=0
                    for i in range(len(crops_count_GT)):
                        if crops_count_GT[i] > 0: # there are true points in the crop
                            if not is_roots_2:
                                state['crops_abs_diff'].append(abs(crops_count_GT[i] - np.round(count_pred[i]))) #count_pred[i]))
                                state['crops_rel_error'].append(abs(crops_count_GT[i] - np.round(count_pred[i])) / crops_count_GT[i])

                                # if config.detect_and_count.type == "both_for_roots":
                                #     crops_abs_diff_TRL.append(abs(crops_TRL_GT[i] - TRL_pred[i])) # count_pred[i]))
                                #     crops_rel_error_TRL.append(abs(crops_TRL_GT[i] - TRL_pred[i]) / crops_count_GT[i])
                                #
                                #     crops_abs_diff_dia.append(abs(crops_dia_GT[i] - dia_pred[i]))  # count_pred[i]))
                                #     crops_rel_error_dia.append(abs(crops_dia_GT[i] - dia_pred[i]) / crops_dia_GT[i])


                            #if config.Detect_and_Estimate.type != "both_for_roots_2":
                                if len(orig_count_GT) > 0:
                                    if orig_count_GT[i] != -1:
                                        state['orig_abs_diff'].append(abs(orig_count_GT[i] - np.round(count_pred[i]))) #count_pred[i]))
                                        state['orig_rel_error'].append(abs(orig_count_GT[i] - np.round(count_pred[i])) / orig_count_GT[i])

                            else:
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

                                            printf('max_overlap =  %.3f', max_overlap_array[i])

                                        else:
                                            print('No gt box with iou>=', config.Detection.iou_threshold)
                                            printf('max_overlap =  %.3f', max_overlap_array[i])

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

                                            printf('max_overlap =   %.3f', max_overlap_array[i])


                                        else:
                                            print('No gt box with iou>=', config.Detection.iou_threshold)
                                            if max_overlap_array is not None:
                                                printf('max_overlap =   %.3f', max_overlap_array[i])
                            print()

                        if to_draw: # drawings of true detections
                            if len(crops_count_GT) > 0:
                                    bbox_crop = sample_anns['img'][i].clone()

                                    # view the crops per image
                                    img2 = Image.open(image_path) # the orig image
                                    imgToVis = img2.copy()

                                    draw2 = PIL.ImageDraw.Draw(img2)

                                    x1 = crops_orig_boxes[i][0]/ scale[0]
                                    y1 = crops_orig_boxes[i][1] / scale[0]
                                    x2 = crops_orig_boxes[i][2] / scale[0]
                                    y2 = crops_orig_boxes[i][3] / scale[0]
                                    draw2.rectangle(((x1, y1), (x2, y2)), outline="red", width=config.DrawProperties.LINE_WIDTH)

                                    if crops_count_GT[i] != 0:
                                        r=2
                                        for p in point_anns:
                                            if p['x'] <= x2 and p['x'] >= x1 and p['y'] <= y2 and p['y'] >= y1:
                                                draw2.ellipse((p['x'] - r, p['y'] - r, p['x'] + r, p['y'] + r),
                                                             fill="black", width=config.DrawProperties.LINE_WIDTH)

                                    if len(gt_boxes)>0:    #len(orig_count_GT) > 0:
                                        #if orig_count_GT[i] != -1:
                                        true_id = int(state['detections_data_any_crop'][image_name]['gt_box_id'][i].item()) # box id starts from 1
                                        if true_id !=-1:

                                            if is_roots_2:
                                                x1 = gt_boxes[true_id-1, 0].item() / scale[0]
                                                y1 = gt_boxes[true_id-1, 1].item() / scale[0]
                                                x2 = gt_boxes[true_id-1, 2].item() / scale[0]
                                                y2 = gt_boxes[true_id-1, 3].item() / scale[0]
                                            else:
                                                filtered = gt_boxes[gt_boxes[:, 5] == true_id][0]
                                                x1 = filtered[0].item() / scale[0]
                                                y1 = filtered[1].item() / scale[0]
                                                x2 = filtered[2].item() / scale[0]
                                                y2 = filtered[3].item() / scale[0]

                                            draw2.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)

                                    # img2.show()
                                    box_name_2 = image_name.split(".jpg")[0] + '_crop_on_image_' + str(i) + '.jpg'  # image_name.split('.jpg')[0]+'_crop_'+str(i)+'.jpg'
                                    img2.save(os.path.join(draw_path, box_name_2))

                                    bbox_img = bbox_crop.cpu().clone().detach()
                                    bbox_img = np.array(255 * unnormalize(bbox_img))
                                    bbox_img[bbox_img < 0] = 0
                                    bbox_img[bbox_img > 255] = 255

                                    bbox_img = np.transpose(bbox_img, (1, 2, 0))

                                    img3 = transforms.ToPILImage(mode='RGB')(bbox_img.astype(np.uint8))
                                    imgToVis = img3.copy()

                                    draw3 = PIL.ImageDraw.Draw(img3)

                                    for j in range(len(relevant_points_anns[i])): #points per box
                                        x = relevant_points_anns[i][j]['x']
                                        y = relevant_points_anns[i][j]['y']
                                        draw3.ellipse(((int(x) - 5, int(y) - 5), (int(x) + 5, int(y) + 5)), fill="black", width=config.DrawProperties.LINE_WIDTH)

                                    box_name_3 = image_name.split(".jpg")[0] + '_crop_' + str(i) + '.jpg'
                                    img3.save(os.path.join(crops_path, box_name_3))

                                    if (config.AttributeEstimation.estimate_type == 'withKeyPoints' and
                                            crops_count_GT[i] != 0): # and config.DrawProperties.DRAW_MAPS): #draw_maps:

                                        true_maps = []
                                        # if config.Detect_and_Estimate.type != "both_for_roots_2":
                                        #     if len(sample_anns['points_annot'][i]) > 1:
                                        #         for p in [1, 2, 3, 4, 5]:
                                        #             true_maps.append(sample_anns['points_annot'][maps_idx][p][0].copy())
                                        #
                                        # else:
                                        for p in [1, 2, 3, 4, 5]:
                                            map = np.array(sample_anns['points_annot'][p][i]).copy()
                                            true_maps.append(map)

                                        pred_maps = []
                                        if len(all_predicted_detection_maps_toDraw[i])>1:
                                            for p in range(5):
                                                pred_maps.append(all_predicted_detection_maps_toDraw[maps_idx][p])

                                        # visualize all maps
                                        if config.DrawProperties.DRAW_MAPS:
                                            for p in [4]: #range(5):
                                                if len(true_maps)>0:
                                                    current_true = true_maps[p]
                                                else:
                                                    current_true = None

                                                if len(pred_maps) > 0:
                                                    current_pred = pred_maps[p].cpu()
                                                else:
                                                    current_pred = None

                                                map_name = image_name.split(".jpg")[0] +'_crop_' + str(i) +'_map_' + str(p+1)
                                                #visualize_pointMaps(current_pred, current_true, map_name, imgToVis, config.DrawProperties.maps_path)
                                                visualize_KeyPointsHeatmaps(current_pred, current_true, image_name.split(".jpg")[0],
                                                                            map_name, imgToVis, config.DrawProperties.maps_path)


                                        # Visualize the fifth map
                                        # if config.DrawProperties.DRAW_MAPS:
                                        #     visualize_images(pred_maps[4], true_maps[0],
                                        #                      image_name + '_crop_' + str(i) + '_map_' + str(4 + 1), imgToVis, draw_path)

                                        # draw for model 2:
                                        # img2.show()
                                        # visualize_images(estimation_outputs[1][i], sample_anns['points_annot'][1][i],
                                        #                  image_name.split('.jpg')[0] + '_' + str(i) + '_' + str(1), dataset,
                                        #                  model, imgToVis)
                                        # for p in range(1,6):
                                        #     visualize_images(estimation_outputs[p][i], sample_anns['points_annot'][p][i],
                                        #                      image_name.split('.jpg')[0]+'_'+str(i)+'_'+str(p), dataset, model, imgToVis)

                                        maps_idx += 1

                    if verbose:
                        if not args.have_GT and len(estimation_outputs) >0 and is_roots_2:
                            for i in range(len(TRL_pred)):
                                printf(
                                    "color_pred: %.3f | TRL_pred: %.3f | dia_pred: %.3f \n",
                                    color_pred[i], TRL_pred[i], dia_pred[i]
                                )
                            print()

        ################################################################################################################
        # Get results summary
        ################################################################################################################

        # obj detection results
        # WRONG----------------------------------------------------------------------------------------------------------------------
        # if args.evaluate_detection:
        #     mAP, precision, recall = objects_recall_precision(state['all_annotations'], state['all_detections'])
        #     print(f'Object detection evaluation: mAP = {mAP:.3f}, precision = {precision[-1]:.3f}, recall = {recall[-1]:.3f}')
        #
        #     # ToDo - check difference in comparison to evaluation with kcsv_2 (in runner)
        #
        #     # print recall and precision to csv
        #     csv_columns = ['recall', 'precision']
        #     csv_file = os.path.join(config.General.files_path, "obj_recall_precision.csv")
        #     f = open(csv_file, 'w', newline='')
        #     with f:
        #         writer = csv.writer(f)
        #         writer.writerow(csv_columns)
        #         for w in range(len(recall)):
        #             myrow = []
        #             myrow.append(recall[w])
        #             myrow.append(precision[w])
        #             writer.writerow(myrow)

        # print('Get results summary:')

        if not is_roots_2:
            if len(state['all_predicted_counts'])==0:
                if verbose:
                    print('There are no images with predicted boxes')
                #return []
            num_of_images = len(state['all_crops_GT_counts'])
            relevant_arr = state['all_crops_GT_counts']

        elif is_roots_2:
            if len(state['all_predicted_TRL'])==0:
                if verbose:
                    print('There are no images with predicted boxes')

            num_of_images = len(state['all_predicted_TRL'])
            relevant_arr = state['all_predicted_TRL']



        # gather all results
        if is_roots_2:
            #total_crops_GT_TRL = []
            total_predicted_TRL = []

            total_orig_GT_TRL = []
            total_predicted_for_orig_boxes_TRL = []
            total_orig_box_for_TRL = 0

            #total_crops_GT_dia = []
            total_predicted_dia = []

            total_orig_GT_dia = []
            total_predicted_for_orig_boxes_dia = []
            total_orig_box_for_dia = 0

        else:
            total_crops_GT_counts = []
            total_predicted_counts = []

            total_orig_GT_counts = []
            total_predicted_for_orig_boxes = []
            total_orig_box_for_count = 0

        for n in range(num_of_images):
            for j in range(len(relevant_arr[n])):

                if is_roots_2:
                    #total_crops_GT_TRL.append(state['all_crops_GT_TRL'][n][j])
                    total_predicted_TRL.append(state['all_predicted_TRL'][n][j])

                    #total_crops_GT_dia.append(state['all_crops_GT_dia'][n][j])
                    total_predicted_dia.append(state['all_predicted_dia'][n][j])

                    if len(state['all_orig_GT_TRL'][n]) > 0:
                        if state['all_orig_GT_TRL'][n][j] != -1:
                            total_orig_GT_TRL.append(state['all_orig_GT_TRL'][n][j])
                            total_orig_box_for_TRL += 1
                            total_predicted_for_orig_boxes_TRL.append(state['all_predicted_TRL'][n][j])

                            total_orig_GT_dia.append(state['all_orig_GT_dia'][n][j])
                            total_orig_box_for_dia += 1
                            total_predicted_for_orig_boxes_dia.append(state['all_predicted_dia'][n][j])

                else:
                    total_crops_GT_counts.append(state['all_crops_GT_counts'][n][j])
                    total_predicted_counts.append(state['all_predicted_counts'][n][j])

                    if len(state['all_orig_GT_counts'][n])>0:
                        if state['all_orig_GT_counts'][n][j] != -1:
                            total_orig_GT_counts.append(state['all_orig_GT_counts'][n][j])
                            total_orig_box_for_count+=1
                            total_predicted_for_orig_boxes.append(state['all_predicted_counts'][n][j])

        if not is_roots_2:
            if len(total_crops_GT_counts)>0:
                total_crop_boxes = len(total_crops_GT_counts)

                crops_avg_abs_count_diff = SumOfAbsDifferences(total_crops_GT_counts, total_predicted_counts) / total_crop_boxes
                crops_avg_rel_error = np.mean(state['crops_rel_error'])
                crops_MSE = np.mean((np.array(total_crops_GT_counts) - np.array(total_predicted_counts)) ** 2)
            else:
                total_crop_boxes = -1
                crops_avg_abs_count_diff = -1
                crops_avg_rel_error = -1
                crops_MSE = -1

            if total_orig_box_for_count>0:
                orig_avg_abs_count_diff = SumOfAbsDifferences(total_orig_GT_counts, total_predicted_for_orig_boxes) / total_orig_box_for_count
                orig_avg_rel_error = np.mean(state['orig_rel_error'])
                orig_MSE = np.mean((np.array(total_orig_GT_counts) - np.array(total_predicted_for_orig_boxes)) ** 2)

            else:
                print('No gt boxes for any image')
                orig_avg_abs_count_diff = -1
                orig_avg_rel_error=-1
                orig_MSE = -1

                #return []

            if total_crop_boxes >0:
                crops_count_agr = 0
                for i in range(total_crop_boxes):
                    if total_crops_GT_counts[i] == total_predicted_counts[i]:
                        crops_count_agr += 1
                crops_count_agr = crops_count_agr / total_crop_boxes

            else:
                crops_count_agr = -1

            if total_orig_box_for_count >0:
                orig_count_agr = 0
                for i in range(total_orig_box_for_count):
                    if total_orig_GT_counts[i] == total_predicted_for_orig_boxes[i]:
                        orig_count_agr += 1
                orig_count_agr = orig_count_agr / total_orig_box_for_count

            else:
                orig_count_agr=-1


            if len(total_crops_GT_counts)> 0:
                crops_var_GT_counts = np.var(total_crops_GT_counts)
                crops_FVU = crops_MSE/crops_var_GT_counts

                orig_var_GT_counts = np.var(total_orig_GT_counts)
                orig_FVU = orig_MSE / orig_var_GT_counts
                orig_mean_GT_counts = np.mean(total_orig_GT_counts)

                precision_det = state['found_orig_objects'] / (state['found_orig_objects'] + state['FP'])

            else:
                crops_var_GT_counts=-1
                crops_FVU=-1

                orig_var_GT_counts = -1
                orig_FVU = -1
                orig_mean_GT_counts = -1

                precision_det = -1

        else:
            if total_orig_box_for_TRL > 0:

                orig_avg_abs_TRL_diff = SumOfAbsDifferences(total_orig_GT_TRL, total_predicted_for_orig_boxes_TRL) / total_orig_box_for_TRL
                orig_avg_rel_error_TRL = np.mean(state['orig_rel_error_TRL'])
                orig_MSE_TRL = np.mean((np.array(total_orig_GT_TRL) - np.array(total_predicted_for_orig_boxes_TRL)) ** 2)

                orig_avg_abs_dia_diff = SumOfAbsDifferences(total_orig_GT_dia,
                                                            total_predicted_for_orig_boxes_dia) / total_orig_box_for_dia
                orig_avg_rel_error_dia = np.mean(state['orig_rel_error_dia'])
                orig_MSE_dia = np.mean((np.array(total_orig_GT_dia) - np.array(total_predicted_for_orig_boxes_dia)) ** 2)

                orig_var_GT_TRL = np.var(total_orig_GT_TRL)
                orig_FVU_TRL = orig_MSE_TRL / orig_var_GT_TRL
                orig_mean_GT_TRL = np.mean(total_orig_GT_TRL)

                orig_var_GT_dia = np.var(total_orig_GT_dia)
                orig_FVU_dia = orig_MSE_dia / orig_var_GT_dia
                orig_mean_GT_dia = np.mean(total_orig_GT_dia)

                precision_det = state['found_orig_objects'] / (state['found_orig_objects'] + state['FP'])

            else:
                if args.have_GT:
                    print('No gt boxes for any image')
                orig_avg_rel_error_TRL = 100000
                orig_var_GT_TRL = -1
                orig_FVU_TRL = -1
                orig_mean_GT_TRL = -1

                orig_var_GT_dia = -1
                orig_FVU_dia = -1
                orig_mean_GT_dia = -1

                precision_det = -1

        if verbose:

            if not is_roots_2:

                printf("====================================================================================================\n")
                printf("Summary - for crops count \n")
                printf("crops_avg_abs_count_diff: %.3f | crops_count_agreement: %.3f | crops_MSE: %.3f | crops_avg_relative_error: %.3f | crops_1-FVU: %3f\n",
                    crops_avg_abs_count_diff, crops_count_agr, crops_MSE, crops_avg_rel_error, 1-crops_FVU)
                print()

                printf("====================================================================================================\n")

                printf("Summary - for gt boxes count \n")
                printf(
                    "orig_avg_abs_count_diff: %.3f | orig_count_agreement: %.3f | orig_MSE: %.3f | orig_avg_relative_error: %.3f | orig_1-FVU: %3f\n",
                    orig_avg_abs_count_diff, orig_count_agr, orig_MSE, orig_avg_rel_error, 1 - orig_FVU)

                printf("====================================================================================================\n")

            else:
                if args.have_GT:

                    printf("====================================================================================================\n")
                    printf("Summary - for gt boxes TRL and dia \n")
                    printf(
                        "orig_avg_abs_TRL_diff: %.3f | orig_MSE_TRL: %.3f | orig_avg_relative_error_TRL: %.3f | orig_1-FVU_TRL: %3f\n",
                        orig_avg_abs_TRL_diff, orig_MSE_TRL, orig_avg_rel_error_TRL, 1 - orig_FVU_TRL)
                    printf(
                        "orig_avg_abs_dia_diff: %.3f | orig_MSE_dia: %.3f | orig_avg_relative_error_dia: %.3f | orig_1-FVU_dia: %3f\n",
                        orig_avg_abs_dia_diff, orig_MSE_dia, orig_avg_rel_error_dia, 1 - orig_FVU_dia)

                    print()

                    print("====================================================================================================\n")
                    printf("General Stats\n")
                    printf("gt_objects_withGTpoints = %d\n", state['gt_objects_withGTpoints'])
                    printf("found_orig_objects_withPoints = %d (%.2f%% of gt objects [recall])\n", state['found_orig_objects'],
                           100 * state['found_orig_objects'] / state['gt_objects_withGTpoints'])
                    printf("precision = %.3f\n\n", precision_det)

            if not is_roots_2:
                printf("====================================================================================================\n")
                printf("GT data summary\n")
                printf("all_orig_avg_GT_counts: %.3f\n",   np.mean(state['all_data_gt_count']))
                printf("all_orig_var_GT_counts: %.3f\n",   np.var(state['all_data_gt_count']))
                printf("all_orig_std_GT_counts: %.3f\n\n", np.sqrt(np.var(state['all_data_gt_count'])))

                printf("====================================================================================================\n")
                printf("Found GT data summary\n")
                printf("orig_avg_GT_counts: %.3f\n",   orig_mean_GT_counts)
                printf("orig_var_GT_counts: %.3f\n",   orig_var_GT_counts)
                printf("orig_std_GT_counts: %.3f\n\n", np.sqrt(orig_var_GT_counts))

            print("=====================================================================================================\n")
            print('Crops stats\n')
            print('num of crops = ', len(state['predicted_counts_any_crop']))
            print('FP = ', state['FP'])
            print('found_orig_objects = ', state['found_orig_objects'])
            #print('matched_without_gt_points = ', state['matched_without_gt_points'])
            print('crops_without_gt_points = ', state['crops_without_gt_points'])

            if not is_roots_2:
                print('summaries to num of crops ?')
                print('avg pred count per crop = {:.2f}'.format(np.mean(state['predicted_counts_any_crop'])))
                print('var of pred count per crop = {:.2f}'.format(np.var(state['predicted_counts_any_crop'])))
                print('avg of per img predictions  = {:.2f}'.format(np.mean(state['per_im_pred_avg'])))
                print("=====================================================================================================\n")

                print('Per image gt stats\n')
                print('avg of image averages = {:.2f}'.format(np.mean(state['per_im_gt_avg'])))
                print('Std of image averages = {:.2f}'.format(np.std(state['per_im_gt_avg'])))
                print("per image avg: {}".format(np.round(state['per_im_gt_avg'], 2)))
                print("=====================================================================================================\n")

                if args.have_GT:
                    print('Per image gt, predicted avg count')
                    abs_error = []
                    for im in state['per_im_pred_dict'].keys():
                        abs_error.append(np.abs(state['per_im_gt_avg_dict'][im]-state['per_im_pred_dict'][im])/state['per_im_gt_avg_dict'][im])
                        print("{}: avg_gt: {:.2f}, avg_pred: {:.2f}, rel_error: {:.2f}".
                              format(im, state['per_im_gt_avg_dict'][im], state['per_im_pred_dict'][im],
                                     abs_error[-1]))

                    avg_per_image_error = np.mean(abs_error)
                    print("Avg of per image rel_error:{:.4f}".format(avg_per_image_error))

            else:
                if args.have_GT:
                    print("=====================================================================================================\n")
                    print('Per image gt stats\n')
                    print('Per image gt TRL (sum of RL), predicted sum TRL')
                    abs_error_TRL = []
                    for im in state['TRL_per_im_pred_dict'].keys():
                        if state['TRL_per_im_gt_sum_dict'][im] > 0:
                            abs_error_TRL.append(np.abs(state['TRL_per_im_gt_sum_dict'][im] - state['TRL_per_im_pred_dict'][im]) / state['TRL_per_im_gt_sum_dict'][im])
                        else:
                            abs_error_TRL.append(-1)

                        print("{}: sum_gt_TRL: {:0.2f}, pred_TRL: {:0.2f}, rel_error_TRL: {:0.2f}".format(im, state['TRL_per_im_gt_sum_dict'][im],
                                                                                                state['TRL_per_im_pred_dict'][im],
                                                                                                abs_error_TRL[-1]))
                    abs_error_TRL_nonZero = [TRL for TRL in abs_error_TRL if TRL > 0]

                    print("Avg of per image rel_error of TRL (for gt>0):{:0.4f}".format(np.mean(abs_error_TRL_nonZero)))
                    print()

                    abs_error_dia = []
                    for im in state['dia_per_im_pred_dict'].keys():
                        if state['dia_per_im_gt_avg_dict'][im] > 0:
                            abs_error_dia.append(
                                np.abs(state['dia_per_im_gt_avg_dict'][im] - state['dia_per_im_pred_dict'][im]) / state['dia_per_im_gt_avg_dict'][im])
                        else:
                            abs_error_dia.append(-1)

                        print("{}: avg_gt_dia: {:0.2f}, avg_pred_dia: {:0.2f}, rel_error_dia: {:0.2f}".format(im,
                                                                                                            state['dia_per_im_gt_avg_dict'][im],
                                                                                                            state['dia_per_im_pred_dict'][im],
                                                                                                            abs_error_dia[-1]))
                    abs_error_dia_nonZero = [dia for dia in abs_error_dia if dia > 0]
                    print("Avg of per image rel_error of dia:{:0.4f}".format(np.mean(abs_error_dia)))

        #export detections info
        if print_to_files and args is not None:
            if is_roots_2:
                csv_columns = ['img','crop' , 'gt_color', 'pred_color', 'label', 'score', 'max_overlap',  'gt_TRL', 'pred_TRL', 'gt_dia','pred_dia']
            else:
                csv_columns = ['img', 'gt_count', 'pred_count', 'label', 'score','max_overlap']
            csv_file = os.path.join(config.General.files_path, "detections_data_any_crop_withEmptyIm.csv") #
            f = open(csv_file, 'w', newline='')
            with f:
                writer = csv.writer(f)
                writer.writerow(csv_columns)
                for img in state['detections_data_any_crop'].keys():
                    mydata = state['detections_data_any_crop'][img]
                    for i in range(len(mydata['score'])):
                        myrow = []
                        myrow.append(img)
                        myrow.append(i)

                        if is_roots_2:
                            myrow.append(mydata['color_gt'][i])
                            myrow.append(mydata['color_pred'][i])
                        else:
                            myrow.append(mydata['gt_count'][i])
                            myrow.append(mydata['pred'][i])

                        myrow.append(mydata['label'][i])
                        myrow.append(mydata['score'][i])
                        myrow.append(mydata['max_overlap'][i])

                        if is_roots_2:
                            myrow.append(mydata['TRL_gt'][i])
                            myrow.append(mydata['TRL_pred'][i])
                            myrow.append(mydata['dia_gt'][i])
                            myrow.append(mydata['dia_pred'][i])

                        writer.writerow(myrow)


            #export not found gt info
            if is_roots_2:
                csv_columns = ['img', 'gt_color', 'pred_color', 'label', 'score', 'max_overlap', 'gt_TRL', 'pred_TRL', 'gt_dia', 'pred_dia']
            else:
                csv_columns = ['img', 'gt_count', 'pred', 'label', 'score', 'max_overlap']
            csv_file = os.path.join(config.General.files_path, "not_found_gt_count.csv")
            f = open(csv_file, 'w', newline='')
            with f:
                writer = csv.writer(f)
                writer.writerow(csv_columns)
                for img in state['not_found_gt'].keys():
                    mydata = state['not_found_gt'][img]
                    if len(mydata)>0:
                        if config.Detect_and_Estimate.type == "both":
                            data_pred = mydata['pred']
                        elif is_roots_2:
                            data_pred = mydata['TRL_pred']

                        for i in range(len(data_pred)):
                            myrow = []
                            myrow.append(img)
                            if is_roots_2:
                                myrow.append(mydata['color_gt'][i])
                                myrow.append(mydata['color_pred'][i])

                            else:
                                myrow.append(mydata['gt_count'][i])
                                myrow.append(mydata['pred'][i])

                            myrow.append(mydata['label'][i])
                            myrow.append(mydata['score'][i])
                            myrow.append(mydata['max_overlap'][i])

                            if is_roots_2:
                                myrow.append(mydata['TRL_gt'][i])
                                myrow.append(mydata['TRL_pred'][i])
                                myrow.append(mydata['dia_gt'][i])
                                myrow.append(mydata['dia_pred'][i])

                            writer.writerow(myrow)
                    else:
                        myrow = []
                        myrow.append(img)
                        myrow.append('None')
                        writer.writerow(myrow)

            # export image data of images without box predictions
            if is_roots_2:
                csv_columns = ['img', 'gt_color', 'pred_color', 'label', 'score','max_overlap', 'gt_TRL', 'pred_TRL', 'gt_dia', 'pred_dia']
            else:
                csv_columns = ['img', 'gt_count', 'pred', 'label', 'score', 'max_overlap']

            csv_file = os.path.join(config.General.files_path, "images_without_detections.csv")  #
            f = open(csv_file, 'w', newline='')
            with f:
                writer = csv.writer(f)
                writer.writerow(csv_columns)
                for img in state['no_predictions'].keys():
                    mydata = state['no_predictions'][img]
                    if config.Detect_and_Estimate.type == "both":
                        data_pred = mydata['pred']
                    elif is_roots_2:
                        data_pred = mydata['TRL_pred']

                    if len(data_pred) > 0:
                        for i in range(len(data_pred)):
                            myrow = []
                            myrow.append(img)
                            if is_roots_2:
                                myrow.append(mydata['color_gt'][i])
                                myrow.append(mydata['color_pred'][i])
                            else:
                                myrow.append(mydata['gt_count'][i])
                                myrow.append(mydata['pred'][i])

                            myrow.append(mydata['label'][i])
                            myrow.append(mydata['score'][i])
                            myrow.append(mydata['max_overlap'][i])

                            if is_roots_2:
                                myrow.append(mydata['TRL_gt'][i])
                                myrow.append(mydata['TRL_pred'][i])
                                myrow.append(mydata['dia_gt'][i])
                                myrow.append(mydata['dia_pred'][i])

                            writer.writerow(myrow)

                    else:
                        myrow = []
                        myrow.append(img)
                        if is_roots_2:
                            myrow.append(-1) #'color_pred'
                            myrow.append(-1) #'color_gt'
                        else:
                            myrow.append(0) # 'pred'
                            myrow.append(0) #'gt_count'

                        myrow.append(-1) #'label'
                        myrow.append(-1) #'score'
                        myrow.append(-1) #'max_overlap'

                        if is_roots_2:
                            myrow.append(0) #'TRL_pred'
                            myrow.append(0) #'TRL_gt'
                            myrow.append(0) #'dia_pred'
                            myrow.append(0) #'dia_gt'

                        writer.writerow(myrow)


        model.train()
        print()
        if config.AttributeEstimation.calc_det_performance and config.AttributeEstimation.estimate_type == 'withKeyPoints':
            recall, precision, ap = calc_points_recall_precision_ap(state['T'], state['P'])
            print(f'Points detection evaluation: mAP = {ap:.3f}, recall = {recall[-1]:.3f}, precision = {precision[-1]:.3f}')
            plot_PR_curve(recall, precision, ap, save_path=config.General.files_path, plots_name = 'Points_PR_curve.png') #config.General.experiment_path)

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

        if not is_roots_2:
            count_points_in_crop = getattr(model, "count_points_in_crop", False)
            final_rel_error = crops_avg_rel_error if count_points_in_crop else orig_avg_rel_error #orig_avg_rel_error #
            out = [final_rel_error,
                   state['gt_objects_withGTpoints'],
                   state['found_orig_objects'],
                   state['found_orig_objects']/state['gt_objects_withGTpoints'],
                   precision_det,
                   orig_avg_abs_count_diff,
                   orig_count_agr,
                   orig_MSE,
                   (1 - orig_FVU)]
        else:
            out = [orig_avg_rel_error_TRL]

        return out









