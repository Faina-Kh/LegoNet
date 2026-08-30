import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


KEYPOINT_CANDIDATE_THRESHOLD = 0.02


def process_keypoint_map_for_evaluation(model, raw_map):
    """Apply the model's trained keypoint-suppression path before evaluation."""
    model_instance = getattr(model, "module", model)
    find_module = getattr(model_instance, "find_2", None)
    if find_module is None:
        find_module = getattr(model_instance, "find", None)
    if find_module is None or not hasattr(find_module, "process_keypoint_map"):
        raise AttributeError(
            "The keypoint model does not expose process_keypoint_map()."
        )
    return find_module.process_keypoint_map(raw_map)


def extract_plant_BB(image_name, activation_map):
    image_shape = activation_map.shape
    mask_image_path = image_name + '_fg.png'
    plant_mask_image = cv2.imread(mask_image_path, 0)

    plant_mask_image = cv2.resize(plant_mask_image, (image_shape[1], image_shape[0]))
    Ys, Xs = np.nonzero(plant_mask_image)
    y_min = np.min(Ys)
    y_max = np.max(Ys)
    x_min = np.min(Xs)
    x_max = np.max(Xs)

    return (x_max - x_min),(y_max - y_min)


def points_detection_t_p(detections_map_1, GT_centers, alpha=0.1): #local_soft_max_activations, image_name, model, image, GT_centers, alpha=0.1):
    # local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
    #                                              layer_name='smooth_step_function2')
    # local_soft_max_activations = local_soft_max_activations[0][0, :, :, 0]
    detections_map = detections_map_1.clone().cpu().numpy()
    detections_map = np.where(
        detections_map > KEYPOINT_CANDIDATE_THRESHOLD,
        detections_map,
        0,
    )

    # plt.hist(detections_map.copy().reshape(detections_map.size))
    # plt.savefig(config.General.save_path+'\\hist.png')

    w_plant, h_plant = detections_map_1.shape # extract_plant_BB(image_name, local_soft_max_activations)
    GT_centers = np.where(GT_centers == 1, GT_centers, 0)
    Y_detect, X_detect = np.nonzero(np.array(detections_map))
    Y_GT, X_GT = np.nonzero(GT_centers)
    detection_scores = np.array(detections_map[Y_detect, X_detect])
    detections = np.array([Y_detect,X_detect, detection_scores])
    sorted_detections = detections[:, (detections[2, :]*-1).argsort()]

    reduced_GT_centers = np.array([Y_GT, X_GT])
    pck_val_thresh = alpha*np.max([w_plant, h_plant])
    t=[]
    p=[]
    for det_num in range(sorted_detections.shape[1]):
        det = sorted_detections[[0,1], det_num]
        if reduced_GT_centers.shape[1] == 0:
            p.append(sorted_detections[-1, det_num])
            t.append(0)
            continue
        dists = np.sqrt(np.sum(np.array([reduced_GT_centers[:, i] - det for i in range(reduced_GT_centers.shape[1])]) ** 2, axis=-1))
        closest_GT_ind = np.argmin(dists)
        min_dist = np.min(dists)
        if(min_dist <= pck_val_thresh):
            p.append(sorted_detections[-1, det_num])
            t.append(1)
            reduced_GT_centers = np.delete(reduced_GT_centers, closest_GT_ind, 1)
        else:
            p.append(sorted_detections[-1, det_num])
            t.append(0)

    if len(reduced_GT_centers):
        for i in range(reduced_GT_centers.shape[1]):
            p.append(0)
            t.append(1)

    return t, p


def measure_ap_forKP(rec, prec):
    # correct AP calculation
    # append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_points_recall_precision_ap(T, P):
    P = np.array(P)
    T = np.array(T)
    npos = np.sum(T)
    # clean zeros (that represent true negative)
    T = T[np.where(P > 0)]
    P = P[np.where(P > 0)]
    # sort by confidence
    sorted_ind = np.argsort(-P)
    P = P[sorted_ind]
    T = T[sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(P)
    if nd > 0:
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for i in np.arange(nd):
            if T[i] == 1 and P[i] > 0:
                tp[i] = 1.
            elif T[i] == 0 and P[i] > 0:
                fp[i] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = (
            tp / float(npos)
            if npos > 0
            else np.zeros_like(tp)
        )
        precision = tp / (tp + fp)
    else:
        recall = [0]
        precision = [0]

    ap = measure_ap_forKP(recall, precision)

    return recall, precision, ap


def _format_heatmap_value(value, attribute_name: str) -> str:
    """Format discrete counts as integers and continuous attributes to 2 decimals."""
    if attribute_name.casefold() == "count":
        return str(int(np.rint(value)))
    return str(np.round(value, 2))


def visualize_KeyPointsHeatmaps(predicted_map, gt_map, image_name, map_name, imgToVis, draw_path, count_pred = None,
                                count_GT=None, font_size = 30, attribute_name = "TRL", attribute_unit = "mm",
                                heatmap_vmax: float | None = None): #"count"

    font = ImageFont.truetype("arial.ttf", font_size) #15) #60

    if gt_map is not None:
        if gt_map.sum() == 0:
            draw_path = os.path.join(draw_path, "no GT objects" )
            os.makedirs(draw_path, exist_ok=True)

    # Draw GT activations:
    img_copy = imgToVis.copy()
    background = img_copy.convert("RGBA")
    BG_w, BG_h = background.size

    background_2 = img_copy.convert('L')   #convert image to monochrome
    background_2.save(draw_path + '/' + image_name + '_background.png')
    background_2 = Image.open(draw_path + '/' + image_name + '_background.png')
    background_2 = background_2.convert("RGBA")

    if gt_map is not None:
        anno = gt_map.copy() #.cpu().numpy().copy()

    if gt_map is not None:
        save_options = (
            {"vmin": 0.0, "vmax": heatmap_vmax}
            if heatmap_vmax is not None
            else {}
        )
        plt.imsave(draw_path + '/' + map_name+ '_anno.png', anno, **save_options)
        gt_anns = Image.open(draw_path + '/' + map_name+ '_anno.png')
        gt_anns = gt_anns.resize((BG_w, BG_h), Image.Resampling.LANCZOS) #Image.ANTIALIAS)
        gt_anns.save(draw_path + '/' + map_name + '_anno.png')

        alphaBlended = Image.blend(gt_anns, background_2, 0.7)

        if count_GT is not None:
            draw = ImageDraw.Draw(alphaBlended)
            draw.text((50, 50), "GT "+ attribute_name+ " = "+_format_heatmap_value(count_GT, attribute_name)+ " " + attribute_unit,
                      (255, 255, 255), font=font)

        alphaBlended.save(draw_path + '/' + map_name+'_Blended_GT.png')

    # Predicted map #######################################################################################################
    if predicted_map is not None:
        predicted_map_path = os.path.join(
            draw_path, map_name + "_predicted_map_tmp.png"
        )
        predicted_heatmap_path = os.path.join(
            draw_path, map_name + "_Predicted.png"
        )
        save_options = (
            {"vmin": 0.0, "vmax": heatmap_vmax}
            if heatmap_vmax is not None
            else {}
        )
        plt.imsave(predicted_map_path, predicted_map, **save_options)
        relu_pred = Image.open(predicted_map_path)

        relu_pred = relu_pred.resize((BG_w, BG_h))

        alphaBlended_relu = Image.blend(relu_pred, background_2.convert('RGBA'), 0.7)

        if count_pred is not None:
            draw = ImageDraw.Draw(alphaBlended_relu)
            draw.text((50, 50), "Predicted "+ attribute_name+ " = " + _format_heatmap_value(count_pred, attribute_name)+ " " +
                      attribute_unit,(255, 255, 255), font=font)

        alphaBlended_relu.save(predicted_heatmap_path)

        os.remove(predicted_map_path)

    if gt_map is not None:
         os.remove(draw_path + '/' + map_name+ '_anno.png')
    os.remove(draw_path + '/' + image_name + '_background.png')

