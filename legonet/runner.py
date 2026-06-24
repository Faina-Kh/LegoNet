import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from legonet.eval import attribute_estimation_eval, detection_eval
from legonet.data_setup import build_data
import random
import gc

from legonet.inference import run_inference
from legonet.legoNet_build import model_build
from legonet import utils
from legonet.checkpointing import save_epoch_checkpoint
from legonet.model_setup import export_legacy_weights, load_requested_weights
from legonet.training_evaluation import (
    evaluate_combined_iou_sweep,
    evaluate_combined_once,
    evaluate_detection,
)


class Tee:
    """Write output to multiple file-like streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_bn(model):
    '''Freeze BatchNorm layers.'''
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def print_args(args, file_path):

    with open(file_path, "w", encoding="utf-8") as f:

        def printf(msg):
            print(msg, end="")   # to screen
            f.write(msg)         # to file

        printf("=====================================================================\n")
        printf("Run Parameters\n")
        printf("=====================================================================\n")

        for var_name, var_val in vars(args).items():
            printf(f"{var_name}: {var_val}\n")

        printf(f"experiment path: {config.General.experiment_path}\n")

        printf("=====================================================================\n\n")




def run(args=None):

    print_args(args, args.txt_results) #utils.print_args(args)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(args.txt_results, "a", encoding="utf-8") as txt_results:
        sys.stdout = Tee(original_stdout, txt_results)
        sys.stderr = Tee(original_stderr, txt_results)
        try:
            return _run(args)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run(args=None):

    # set seeders
    torch.manual_seed(19860318)
    np.random.seed(19830614)
    random.seed(0)

    data = build_data(args)
    dataset_train = data.dataset_train
    dataset_val = data.dataset_val
    sampler = data.sampler
    sampler_val = data.sampler_val
    dataloader_train = data.dataloader_train
    dataloader_val = data.dataloader_val


    # build the model
    legonet = model_build(args, dataset_train, dataset_val)

    if args.load_weights:
        load_requested_weights(legonet, args)

    elif args.save_from_model_file:
        export_legacy_weights(legonet, args)
        return

    if (args.network_type == "bbox_detection" or args.network_type == "both" or args.network_type == "both_for_roots_2"
            or args.network_type == "both_Back2bFind2b"):
        if args.freeze_detection:
            legonet.freeze_detector()

    legonet = legonet.to(config.General.device)

    if args.run_script=='Training':

        legonet.training = True

        optimizer = optim.Adam(legonet.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, eps=0.0001)
        loss_hist = collections.deque(maxlen=500)


        print('Num training images: {}'.format(len(dataset_train)))

        best_rel_error = 100.0
        best_mAP = 0.0
        best_avg_rel_error = 100.0
        for epoch_num in range(args.epochs):

            legonet.train()
            legonet.freeze_bn() # freeze_bn(unwrap_model(legonet)) #legonet.freeze_bn()

            if (args.network_type == 'bbox_detection' or args.network_type == 'both' or
                args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b"):
                legonet.freeze_detector()

            epoch_loss = []
            epoch_loss_per_task = []

            if args.network_type == 'bbox_detection':
                epoch_loss_per_task = {
                    "classification": [],
                    "regression": []
                }
            else:
                if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                    if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "color": [],
                            "maps": [],
                            "length": [],
                            "diameter": []
                        }

                    elif  args.network_type == 'counting_lean':
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "l1_estimation": [], #l1
                            "maps": []
                        }

                    elif args.network_type == 'both':
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "l1_counting": [],
                            "maps": []
                        }

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    if args.network_type == "both_for_roots_2":
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "color": [],
                            "length": [],
                            "diameter": []
                        }

                    elif args.network_type == "both":
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "counting": []
                        }

                    elif args.network_type == "counting_reg":
                        epoch_loss_per_task = {
                            "reg_estimation": []
                        }

            for iter_num, data in enumerate(dataloader_train):
                try:
                    optimizer.zero_grad()

                    if args.network_type == 'bbox_detection':
                        bbox_classification_loss, bbox_regression_loss = legonet([data['img'].to(config.General.device).float(),
                                                                                  data['bbox_annot'].to(config.General.device)])

                        bbox_classification_loss = bbox_classification_loss.mean()
                        bbox_regression_loss = bbox_regression_loss.mean()

                        loss = bbox_classification_loss + bbox_regression_loss

                        bbox_classification_loss_value = bbox_classification_loss.item()
                        bbox_regression_loss_value = bbox_regression_loss.item()

                    elif args.network_type == 'counting_lean':
                        l1_estimation_loss, maps_loss = legonet([data['img'].to(config.General.device).float(), data['annot']])
                        l1_estimation_loss = args.loss_weight*l1_estimation_loss

                        if l1_estimation_loss is not None and maps_loss is not None:
                            l1_estimation_loss = l1_estimation_loss.mean()
                            maps_loss = maps_loss.mean()
                            loss = l1_estimation_loss + maps_loss

                            l1_estimation_value = l1_estimation_loss.item()
                            maps_loss_value = maps_loss.item()

                        else:
                            l1_estimation_value = -1
                            maps_loss_value = -1
                            loss = None

                    elif args.network_type == 'counting_reg':
                        reg_estimation_loss = legonet([data['img'].to(config.General.device).float(), data['annot']])
                        reg_estimation_loss = args.loss_weight * reg_estimation_loss
                        reg_estimation_loss = reg_estimation_loss.mean()
                        loss = reg_estimation_loss

                        reg_estimation_loss_value = reg_estimation_loss.item()

                    elif (args.network_type == 'both' or args.network_type == "both_for_roots_2"
                          or args.network_type == "both_Back2bFind2b"):

                        if torch.cuda.is_available():
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                                if args.network_type == 'both':
                                    bbox_classification_loss, bbox_regression_loss, l1_counting_loss, maps_loss = \
                                        legonet([data['img'].to(config.General.device).float(),
                                                 [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                                 torch.tensor(sampler.groups[iter_num])])

                                elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                    if 'points_annot' in data.keys():
                                        bbox_classification_loss, bbox_regression_loss, color_loss, maps_loss, length_loss, diameter_loss=\
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                                     torch.tensor(sampler.groups[iter_num])])

                                    else:
                                        bbox_classification_loss, bbox_regression_loss, color_loss, maps_loss, length_loss, diameter_loss=\
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), None],
                                                     torch.tensor(sampler.groups[iter_num])])

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                if args.network_type == 'both':
                                    bbox_classification_loss, bbox_regression_loss, reg_counting_loss =  legonet(
                                            [data['img'].to(config.General.device).float(),
                                             [data['bbox_annot'].to(config.General.device),data['points_annot']],
                                             torch.tensor(sampler.groups[iter_num])])

                                elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                    if 'points_annot' in data.keys():
                                        bbox_classification_loss, bbox_regression_loss, color_loss, length_loss, diameter_loss = \
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                                     torch.tensor(sampler.groups[iter_num]), args.do_counting])
                                    else:
                                        bbox_classification_loss, bbox_regression_loss, color_loss, length_loss, diameter_loss = \
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), None],
                                                     torch.tensor(sampler.groups[iter_num]), args.do_counting])

                        else:
                            print(f"Iteration: {iter_num} | CUDA not available")

                        if bbox_classification_loss is not None:
                            bbox_classification_loss = bbox_classification_loss.mean()
                            bbox_regression_loss = bbox_regression_loss.mean()
                            bbox_detection_loss = bbox_classification_loss + bbox_regression_loss

                            bbox_classification_loss_value = bbox_classification_loss.item()
                            bbox_regression_loss_value = bbox_regression_loss.item()
                            detection_loss_value = bbox_classification_loss_value + bbox_regression_loss_value
                        else:
                            bbox_detection_loss = None
                            bbox_classification_loss_value = -1
                            bbox_regression_loss_value = -1
                            detection_loss_value = -1

                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                            if args.network_type == 'both':
                                if l1_counting_loss is not None and maps_loss is not None:
                                    l1_counting_loss = l1_counting_loss.mean()
                                    maps_loss = maps_loss.mean()

                                    l1_counting_loss_value = l1_counting_loss.item()
                                    maps_loss_value = maps_loss.item()

                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss + l1_counting_loss + maps_loss
                                    else:
                                        loss = l1_counting_loss + maps_loss

                                else:
                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss
                                    else:
                                        loss = None

                            if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and maps_loss is not None and length_loss is not None and diameter_loss is not None:
                                    diameter_loss = args.dia_loss_weight * diameter_loss
                                    color_loss = args.color_loss_weight * color_loss  #args.loss_weight * color_loss
                                    maps_loss = args.maps_loss_weight * maps_loss
                                    attributes_estimation_loss = color_loss + maps_loss + length_loss + diameter_loss
                                    attributes_estimation_loss = attributes_estimation_loss.mean()

                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss + attributes_estimation_loss
                                    else:
                                        loss = attributes_estimation_loss

                                    color_loss_value = color_loss.item()
                                    maps_loss_value = maps_loss.item()
                                    length_loss_value = length_loss.item()
                                    diameter_loss_value = diameter_loss.item()

                                else:
                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss
                                    else:
                                        loss = None

                                    attributes_estimation_loss = None
                                    attibutes_estimation_loss_value = -1
                                    color_loss_value = -1
                                    maps_loss_value = -1
                                    length_loss_value = -1
                                    diameter_loss_value = -1

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':

                            if args.network_type == 'both':
                                if reg_counting_loss is not None:
                                    reg_counting_loss = reg_counting_loss.mean()
                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss + reg_counting_loss
                                    else:
                                        loss = reg_counting_loss
                                    reg_counting_loss_value = reg_counting_loss.item()
                                else:
                                    loss = bbox_detection_loss

                            elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and length_loss is not None and diameter_loss is not None:
                                    diameter_loss = args.dia_loss_weight * diameter_loss
                                    color_loss = args.color_loss_weight * color_loss  # args.loss_weight * color_loss
                                    attributes_estimation_loss = color_loss + length_loss + diameter_loss
                                    attributes_estimation_loss = attributes_estimation_loss.mean()

                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss + attributes_estimation_loss
                                    else:
                                        bbox_detection_loss = None
                                        loss = attributes_estimation_loss

                                    color_loss_value = color_loss.item()
                                    length_loss_value = length_loss.item()
                                    diameter_loss_value = diameter_loss.item()

                                else:
                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss
                                    else:
                                        loss = None

                                    attributes_estimation_loss = None
                                    color_loss_value = -1
                                    length_loss_value = -1
                                    diameter_loss_value = -1


                    if bool(loss == 0):
                        continue

                    if loss is not None:
                        if loss.grad_fn is not None:
                            loss.backward()
                    else:
                        continue

                    torch.nn.utils.clip_grad_norm_(legonet.parameters(), 0.1)

                    optimizer.step()

                    loss_hist.append(loss.item())
                    epoch_loss.append(loss.item())

                    if (args.network_type == 'bbox_detection' or args.network_type == 'both'
                            or args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b"):
                        epoch_loss_per_task["classification"].append(bbox_classification_loss_value)
                        epoch_loss_per_task["regression"].append(bbox_regression_loss_value)

                    if not config.General.NETWORK_TYPE == config.NetworkType.detection:
                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                            if args.network_type == 'counting_lean':
                                if l1_estimation_loss is not None and maps_loss is not None:
                                    epoch_loss_per_task["l1_estimation"].append(l1_estimation_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)

                            elif args.network_type == 'both':
                                if l1_counting_loss is not None and maps_loss is not None:
                                    epoch_loss_per_task["l1_counting"].append(l1_counting_loss_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)

                            elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if (color_loss is not None and maps_loss is not None and length_loss is not None and
                                        diameter_loss is not None):
                                    epoch_loss_per_task["color"].append(color_loss_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)
                                    epoch_loss_per_task["length"].append(length_loss_value)
                                    epoch_loss_per_task["diameter"].append(diameter_loss_value)

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':

                            if args.network_type == "both_for_roots_2":
                                if color_loss is not None and length_loss is not None and diameter_loss is not None:
                                    epoch_loss_per_task["color"].append(color_loss_value)
                                    epoch_loss_per_task["length"].append(length_loss_value)
                                    epoch_loss_per_task["diameter"].append(diameter_loss_value)

                            elif args.network_type == 'both':
                                if reg_counting_loss is not None:
                                    epoch_loss_per_task["counting"].append(reg_counting_loss_value)

                            elif args.network_type == 'counting_reg':
                                if reg_estimation_loss is not None:
                                    epoch_loss_per_task["reg_estimation"].append(reg_estimation_loss_value)

                    if args.network_type == 'bbox_detection':
                        print(
                            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(bbox_classification_loss), float(bbox_regression_loss),np.mean(loss_hist)))

                        del bbox_regression_loss
                        del bbox_classification_loss

                    elif args.network_type == 'counting_lean':
                        print(
                            'Epoch: {} | Iteration: {} | l1 loss: {:1.5f} |  maps loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(l1_estimation_value), float(maps_loss_value), np.mean(loss_hist)))

                        del l1_estimation_loss
                        del maps_loss

                    elif args.network_type == 'counting_reg':
                        print(
                            'Epoch: {} | Iteration: {} | estimation loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(reg_estimation_loss_value), np.mean(loss_hist)))

                        del reg_estimation_loss

                    if (args.network_type == 'both' or args.network_type == 'both_for_roots_2'
                            or args.network_type == "both_Back2bFind2b"):
                        if args.network_type == 'both':
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                                print(
                                     'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | '
                                     'KeyPointsMaps loss: {:1.5f} | '
                                     'Counting loss: {:1.5f}'.format(epoch_num, iter_num, detection_loss_value,
                                                                     maps_loss_value, l1_counting_loss_value))

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | '
                                    'Counting loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value, reg_counting_loss_value))

                        if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | '
                                    'KeyPointsMaps loss: {:1.5f} | '
                                    'Length loss: {:1.5f} | Diameter loss: {:1.5f} | Color loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value,
                                        maps_loss_value, length_loss_value, diameter_loss_value, color_loss_value))

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | Length loss: {:1.5f} | '
                                    'Diameter loss: {:1.5f} | Color loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value,
                                        length_loss_value, diameter_loss_value, color_loss_value))

                        del bbox_regression_loss
                        del bbox_classification_loss

                        if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                            if args.network_type == "both":
                                if l1_counting_loss is not None and maps_loss is not None:
                                    del l1_counting_loss
                                    del maps_loss

                            if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and maps_loss is not None and length_loss is not None and diameter_loss is not None:
                                    del attributes_estimation_loss
                                    del color_loss
                                    del maps_loss
                                    del length_loss
                                    del diameter_loss

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                            if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and length_loss is not None and diameter_loss is not None:
                                    del color_loss
                                    del length_loss
                                    del diameter_loss
                                    del attributes_estimation_loss

                            elif args.network_type == "both":
                                del reg_counting_loss

                    gc.collect()
                    torch.cuda.empty_cache()


                except Exception as error:
                    raise RuntimeError(
                        f"Training failed at epoch {epoch_num}, iteration {iter_num}, "
                        f"network type {args.network_type}."
                    ) from error

            if args.network_type == 'bbox_detection':

                utils.printf('Epoch %d summary: classification mean %.5f, regression mean %.5f\n',
                        epoch_num,
                        np.mean(epoch_loss_per_task["classification"]),
                        np.mean(epoch_loss_per_task["regression"]))

                if args.dataset_type == 'kcsv' and args.kcsv_val is not None:

                    if args.eval_in_train:

                        legonet.eval()

                        utils.printf("Evaluating Dataset: ")
                        mAP, precision , recall = detection_eval.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val, legonet,
                                                                score_threshold=config.Detection.min_score,
                                                                iou_threshold=config.Detection.iou_threshold)
                        print(f'Current mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n')
                        with open(args.txt_results, 'a') as f:
                            f.write(f'Epoch: {epoch_num}, mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n')

                        if mAP > best_mAP:
                            best_mAP = mAP
                            print(f'Current best*: {best_mAP:.3f}\n')
                            save_epoch_checkpoint(
                                legonet,
                                epoch_num,
                                replace_existing=True,
                            )

                    elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        save_epoch_checkpoint(legonet, epoch_num)

                elif args.dataset_type ==  "roots_json" :

                    if args.evaluate_detection:
                        legonet.eval()
                        mAP, _, _ = detection_eval.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val, legonet,
                                                             score_threshold=config.Detection.min_score, iou_threshold=config.Detection.iou_threshold) #kcsv_eval_2
                        print(f'Current mAP = {best_mAP:.3f}\n')
                        # if len(precision)==0:
                        #     print('mAP: {:.3f} | precision: None | recall: None | prev_best_mAP: {:.3f} \n'.format(mAP, best_mAP))
                        # else:
                        #     print('mAP: {:.3f} | precision: {:.3f} | recall : {:.3f} | prev_best_mAP: {:.3f} \n'.format(mAP, precision[0], recall[0], best_mAP))

                        if mAP > best_mAP:
                            best_mAP = mAP
                            print(f'Current best*: {best_mAP:.3f}\n')
                            save_epoch_checkpoint(
                                legonet,
                                epoch_num,
                                replace_existing=True,
                            )

            elif args.network_type == 'counting_lean' or args.network_type == 'counting_reg':
                if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    if args.network_type == 'counting_lean':
                        utils.printf(
                            'Epoch %d summary: l1_estimation_loss %.5f, maps_loss %.5f\n',
                            epoch_num,
                            #np.mean(epoch_loss_per_task["counting"]),
                            np.mean(epoch_loss_per_task["l1_estimation"]),
                            np.mean(epoch_loss_per_task["maps"]))

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    utils.printf(
                        'Epoch %d summary: counting mean loss %.5f \n',
                        epoch_num,
                        np.mean(epoch_loss_per_task["reg_estimation"]))


                if args.eval_in_train:

                    legonet.eval()

                    utils.printf("Counting evaluation: ")

                    rel_error = attribute_estimation_eval.eval(dataloader_val, dataset_val, legonet, args)
                    #utils.printf("rel error: %.3f \n", rel_error)
                    print('Rel_error: {:.3f} | prev_best: {:.3f} \n'.format(rel_error, best_rel_error))

                    if rel_error < best_rel_error:
                        best_rel_error = rel_error
                        print(f'Current best*: {best_rel_error:.3f}\n')
                        save_epoch_checkpoint(
                            legonet,
                            epoch_num,
                            replace_existing=True,
                        )

                elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        save_epoch_checkpoint(legonet, epoch_num)

            elif (args.network_type == 'both' or args.network_type == "both_for_roots_2"
                  or args.network_type == "both_Back2bFind2b"):
                if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                    if args.network_type == 'both':
                        utils.printf('Epoch %d summary: classification mean %.5f, regression mean %.5f, '
                                    'l1_counting_loss %.5f, maps_loss %.5f\n',
                                    epoch_num,
                                    np.mean(epoch_loss_per_task["classification"]),
                                    np.mean(epoch_loss_per_task["regression"]),
                                    #np.mean(epoch_loss_per_task["counting"]),
                                    np.mean(epoch_loss_per_task["l1_counting"]),
                                    np.mean(epoch_loss_per_task["maps"]))

                    if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                        utils.printf(
                            'Epoch %d summary: classification mean %.5f, regression mean %.5f, maps_loss %.5f, count_loss %.5f, length_loss %.5f, diameter_loss %.5f\n',
                            epoch_num,
                            np.mean(epoch_loss_per_task["classification"]),
                            np.mean(epoch_loss_per_task["regression"]),
                            np.mean(epoch_loss_per_task["maps"]),
                            np.mean(epoch_loss_per_task["color"]),
                            np.mean(epoch_loss_per_task["length"]),
                            np.mean(epoch_loss_per_task["diameter"])
                            )

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    if args.network_type == "both_for_roots_2":
                        utils.printf(
                            'Epoch %d summary: classification mean %.5f, regression mean %.5f, color_loss %.5f, length_loss %.5f, diameter_loss %.5f\n',
                            epoch_num,
                            np.mean(epoch_loss_per_task["classification"]),
                            np.mean(epoch_loss_per_task["regression"]),
                            np.mean(epoch_loss_per_task["color"]),
                            np.mean(epoch_loss_per_task["length"]),
                            np.mean(epoch_loss_per_task["diameter"])
                        )
                    else:
                        utils.printf(
                            'Epoch %d summary: classification mean %.5f, regression mean %.5f, counting mean %.5f \n',
                            epoch_num,
                            np.mean(epoch_loss_per_task["classification"]),
                            np.mean(epoch_loss_per_task["regression"]),
                            np.mean(epoch_loss_per_task["counting"]))

                print()

                if args.eval_in_train:
                    if args.evaluate_detection:
                        print()
                        print("Object detection evaluation: ")

                        metrics = evaluate_detection(
                            dataset_val,
                            dataloader_val,
                            sampler_val,
                            legonet,
                        )
                        print(
                            f"mAP = {metrics.mean_average_precision:.3f}, "
                            f"precision = {metrics.precision:.3f}, "
                            f"recall = {metrics.recall:.3f}"
                        )

                    print()
                    if torch.cuda.is_available():

                        if args.choose_epoch_by_IoUavg:
                            print(
                                f"{args.network_type} evaluation with min_score = "
                                f"{config.Detection.min_score}: \n"
                            )
                            sweep = evaluate_combined_iou_sweep(
                                dataset_val,
                                dataloader_val,
                                sampler_val,
                                legonet,
                                args,
                                config.Detection.iou_threshold_list,
                            )
                            for measurement in sweep.measurements:
                                print(f"iou_threshold = {measurement.iou_threshold}: \n")
                                displayed_error = (
                                    measurement.relative_error
                                    if measurement.relative_error is not None
                                    else -1
                                )
                                utils.printf("rel error: %.3f\n", displayed_error)
                                print()

                            avg_rel_error = sweep.average_relative_error
                            if avg_rel_error is not None:
                                if avg_rel_error < best_avg_rel_error:
                                    best_avg_rel_error = avg_rel_error
                                    print(f'Current best avg error*: {best_avg_rel_error:.3f}\n')
                                    save_epoch_checkpoint(
                                        legonet,
                                        epoch_num,
                                        replace_existing=True,
                                    )

                        else:
                            print(f'{args.network_type} evaluation, iou_threshold = {config.Detection.iou_threshold}, '
                                  f'min_score = {config.Detection.min_score}: ')

                            rel_error = evaluate_combined_once(
                                dataset_val,
                                dataloader_val,
                                sampler_val,
                                legonet,
                                args,
                            )
                            if rel_error is not None:
                                if rel_error < best_rel_error:
                                    best_rel_error = rel_error
                                    print(f'Current best*: {best_rel_error:.3f}\n')
                                    save_epoch_checkpoint(
                                        legonet,
                                        epoch_num,
                                        replace_existing=True,
                                    )
                            else:
                                rel_error = -1

                            utils.printf("rel error: %.3f\n", rel_error)
                            print()

                    else:
                        print(f"Iteration: {iter_num} | CUDA not available")

                elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        save_epoch_checkpoint(legonet, epoch_num)

            scheduler.step(np.mean(epoch_loss))

        legonet.eval()

    else:
        run_inference(
            args,
            dataset_val,
            dataloader_val,
            sampler_val,
            legonet,
        )

