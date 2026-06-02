import os
import collections
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import config #config_new
from legonet.eval import coco_eval, counting_eval, kcsv_eval_2
from legonet.eval import both_eval_new_241 as both_eval
from legonet.my_dataloader import (
    KCSVDataset,
    CocoDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    Augmenter,
    Normalizer,
    UnNormalizer,
    csv_LCCDataset,
    LCC_collater,
    kcsv_collater,
)

from PIL import Image, ImageDraw, ImageFont
import random
import gc

from legonet.legoNet_build import model_build
from manage_weights import list_checkpoint_modules, load_submodule_weights, save_partial_weights, print_module_names
from legonet import utils


def remove_prevEpoch():
    dir_files = os.listdir(config.General.weights_dir)
    if len(dir_files) > 0:
        current_weights_file = dir_files[0]
        os.remove(os.path.join(config.General.weights_dir, current_weights_file))


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_bn(model):
    '''Freeze BatchNorm layers.'''
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def vis_bbox(dataloader_val,sampler_val,dataset_val, model, unnormalize):
    font = ImageFont.truetype('arial.ttf', 14)
    print()
    #print('Per image inference time with visualization:')
    for idx, data in enumerate(dataloader_val):

        group_idx = sampler_val.groups[idx]
        img_id = dataset_val.image_ids[group_idx[0]]

        image_name = dataset_val.img_info[img_id]['name']

        with torch.no_grad():
            st = time.time()
            if config.General.NETWORK_TYPE == config.NetworkType.detection:
                detection_outputs = model([data['img'].to(config.General.device).float(), [data['bbox_annot'], None],
                                             None]) #, False
            elif config.General.NETWORK_TYPE == config.NetworkType.detection_and_counting:
                if 'points_annot' in data.keys():
                    anns = [data['bbox_annot'], data['points_annot']]
                else:
                    anns = [data['bbox_annot'], None]

                detection_outputs, count_outputs, count_sample, relevant_points, crops_orig_boxes = \
                    model([data['img'].to(config.General.device).float(), anns,
                           torch.tensor(group_idx)])

            scores, classification, transformed_anchors = detection_outputs

            #print('Elapsed time: {:.3f} seconds'.format(time.time()-st))

            idxs = np.where(scores.cpu() > config.Detection.min_score)
            img_array = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img_array[img_array < 0] = 0
            img_array[img_array > 255] = 255

            img_array = np.transpose(img_array, (1, 2, 0))
            # img_array = np.transpose(img_array, (2, 1, 0))

            img = Image.fromarray(np.uint8(img_array))

            draw = ImageDraw.Draw(img)

            # draw predictions
            for j in range(idxs[0].shape[0]):
                ann = transformed_anchors[idxs[0][j], :]
                x1 = int(ann[0])
                y1 = int(ann[1])
                x2 = int(ann[2])
                y2 = int(ann[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                score = scores[idxs[0][j]]

                draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=config.DrawProperties.LINE_WIDTH)
                # draw.text((x1, y2-20), "score = {:.3f}".format(score.item()), font=font)

            # draw GT
            annots = data["bbox_annot"].numpy()[0]
            for ann in annots:
                if ann[0]!= -1:
                    x1 = int(ann[0])
                    y1 = int(ann[1])
                    x2 = int(ann[2])
                    y2 = int(ann[3])
                    label_name = dataset_val.labels[int(ann[4])]

                    draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=config.DrawProperties.LINE_WIDTH)
                    # draw.text((x1, y1+15), label_name, font=font)

            # create image with size (100,100) and black background
            button_img = Image.new('RGBA', (200, 20), "black")

            # put text on image
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((0, 0), image_name, font=font)

            # put button on source image in position (0, 0)
            img.paste(button_img, (0, 0))

            #img.show()
            img.save(os.path.join(config.DrawProperties.save_img_path, image_name.split('.jpg')[0] + "_annot.jpg"))


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

    # set seeders
    torch.manual_seed(19860318)
    np.random.seed(19830614)
    random.seed(0)

    # Create the data loaders
    if args.dataset_type == 'coco':

        dataset_train = CocoDataset(args.dataset_path, set_name='train',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(ann_type='bbox')]))
        dataset_val = CocoDataset(args.dataset_path, set_name='val',
                                  transform=transforms.Compose([Normalizer(), Resizer(ann_type='bbox')]))

        args.collater = collater

    elif (args.dataset_type == 'kcsv' or
          (args.dataset_type == "roots_json" and (args.network_type == "both_for_roots_2" or
                                                  args.network_type == "bbox_detection" or
                                                  args.network_type == "both_Back2bFind2b"))):

        # if args.kcsv_train is None:
        #     raise ValueError('Must provide --csv_train for training purposes')
        #
        # if args.kcsv_classes is None:
        #     raise ValueError('Must provide --csv_classes for training purposes')

        if args.run_script == "Training":
            if args.dataset_type == 'kcsv':
                dataset_train = KCSVDataset(input_file=args.kcsv_train, class_list=args.kcsv_classes,
                                            pre_process = args.pre_process,
                                            transform=transforms.Compose([Normalizer(pre_process = args.pre_process),
                                                                         Resizer(min_side=800, max_side=1333)]))

            elif args.dataset_type == "roots_json":
                dataset_train = KCSVDataset(input_file= args.train_json_file,
                                            transform=transforms.Compose([Normalizer(pre_process=args.pre_process),
                                                                          Resizer(min_side=800, max_side=1333)]),
                                            dataset_type = "roots_json")

        else:
            dataset_train = None


        if args.dataset_type == 'kcsv':
            dataset_val = KCSVDataset(input_file=args.val_file, class_list=args.kcsv_classes,
                                      pre_process = args.pre_process,
                                      transform = transforms.Compose([Normalizer(pre_process = args.pre_process),
                                                                   Resizer(min_side=800, max_side=1333)]))

        elif args.dataset_type == "roots_json":
            dataset_val = KCSVDataset(input_file=args.val_json_file,
                                      transform=transforms.Compose([Normalizer(pre_process=args.pre_process),
                                                                    Resizer(min_side=800, max_side=1333)]),
                                      dataset_type="roots_json",
                                      base_dir= args.base_dir,  #args.dataset_path
                                      have_GT = args.have_GT)


        #current_dataset = dataset_val #dataset_val
        # if args.dataset_name == 'grapes':
        #     per_type = {'CDY': {'img': [], 'boxes': [], 'avg_counts': [], 'std_counts': [], 'total_parts': 0},
        #                 'CFR': {'img': [], 'boxes': [], 'avg_counts': [], 'std_counts': [], 'total_parts': 0},
        #                 'CSV': {'img': [], 'boxes': [], 'avg_counts': [], 'std_counts': [], 'total_parts': 0},
        #                 'SVB': {'img': [], 'boxes': [], 'avg_counts': [], 'std_counts': [], 'total_parts': 0},
        #                 'SYH': {'img': [], 'boxes': [], 'avg_counts': [], 'std_counts': [], 'total_parts': 0}}
        #
        #     total_objects = 0
        #     total_parts = 0
        #     for im in current_dataset.image_data_bbox:
        #         grape_type = im.split("_")[0]
        #         per_im_boxes = 0
        #         per_im_counts = 0
        #
        #         for b in current_dataset.image_data_bbox[im]:
        #             if 'points_count' in b.keys():
        #                 per_im_boxes += 1
        #                 total_objects += 1
        #
        #                 per_im_counts += b['points_count']
        #                 total_parts += b['points_count']
        #
        #         per_type[grape_type]['total_parts'] += per_im_counts
        #
        #         if per_im_boxes > 0:
        #             per_type[grape_type]['avg_counts'].append(per_im_counts / per_im_boxes)
        #             per_type[grape_type]['std_counts'].append(np.std(per_im_counts))
        #             per_type[grape_type]['boxes'].append(per_im_boxes)
        #             per_type[grape_type]['img'].append(im)
        #
        #     for type in per_type.keys():
        #         per_type[type]['type_avg_count'] = np.mean(per_type[type]['avg_counts'])
        #         per_type[type]['type_std_count'] = np.std(per_type[type]['avg_counts'])
        #
        #     csv_columns = ['type', 'count_avg', 'count_std', 'objects',
        #                    'parts']  # ['type','img', 'boxes_num', 'count_avg', 'std']
        #     csv_file = os.path.join(myExpResultsPath, 'KK_Exp_Results_last', "grapes data.csv")
        #     f = open(csv_file, 'w', newline='')
        #     with f:
        #         writer = csv.writer(f)
        #         writer.writerow(csv_columns)
        #         for type in per_type.keys():
        #             myrow = []
        #             myrow.append(type)
        #             myrow.append(per_type[type]['type_avg_count'])
        #             myrow.append(per_type[type]['type_std_count'])
        #             myrow.append(np.sum(per_type[type]['boxes']))
        #             myrow.append(per_type[type]['total_parts'])
        #
        #             writer.writerow(myrow)
        #             # mydata = per_type[type]
        #             # for i in range(len(per_type[type]['img'])):
        #             #     myrow = []
        #             #     myrow.append(type)
        #             #     myrow.append(mydata['img'][i])
        #             #     myrow.append(mydata['boxes'][i])
        #             #     myrow.append(mydata['avg_counts'][i])
        #             #     myrow.append(mydata['std_counts'][i])
        #             #
        #             #     writer.writerow(myrow)

        args.collater = kcsv_collater

    elif args.dataset_type == 'csv_LCC': # or (args.dataset_type == "roots_json" and (args.network_type == "counting_lean_multiple_out" or args.network_type == "counting_lean_multiple_out_V2")):

        if args.run_script == 'Training':

            dataset_train = csv_LCCDataset(
                args.train_csv_leaf_number_file,
                args.train_csv_leaf_location_file,
                pre_process='keras_like',
                ann_type='count',
                transform=transforms.Compose([Normalizer(pre_process = args.pre_process),
                                              Resizer(ann_type='count', min_side=800, max_side=1333)]),
                json_file = args.train_json_file)

        else:
            dataset_train = None


        dataset_val = csv_LCCDataset(
            args.val_csv_leaf_number_file,
            args.val_csv_leaf_location_file,
            pre_process='keras_like',
            ann_type = 'count',
            transform=transforms.Compose([Normalizer(pre_process = args.pre_process),
                                          Resizer(ann_type='count', min_side=800, max_side=1333)]),
            json_file = args.val_json_file,
            base_dir= args.base_dir, #args.dataset_path,
            have_GT= args.have_GT)

        args.collater = LCC_collater

    else:
        raise ValueError('Dataset type not understood (must be csv_LCC or coco), exiting.')

    if args.run_script == 'Training':
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=args.num_workers, collate_fn=args.collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False, do_shuffle=False)
        dataloader_val = DataLoader(dataset_val, num_workers=args.num_workers, collate_fn=args.collater, batch_sampler=sampler_val)


    # build the model
    legonet = model_build(args, dataset_train, dataset_val)

    # Loading weights
    if args.load_weights:
        print('Loading weights from: ', os.path.join(args.myExpPath, "Weights \n"))

        print("All available modules in legoNet: ")
        print_module_names(legonet)

        if args.load_partial_weights:

            if args.load_bbox_det_weights:
                bbox_det_state_dict = torch.load(args.bbox_detection_weights_file, map_location=config.General.device)
                print("Available modules in bbox_detection weights file:", list_checkpoint_modules(bbox_det_state_dict))

                if args.network_type == 'bbox_detection':
                    legonet.load_state_dict(bbox_det_state_dict, strict=False) #strict=False
                    load_submodule_weights(legonet, bbox_det_state_dict,
                                           submodule_names=['backbone_1', 'find_1', 'where'], strict=False) #strict=False

                elif (args.network_type == 'both' or args.network_type == "both_for_roots_2"
                      or args.network_type == "both_Back2bFind2b"): #config.General.MODE == 'Training' and  (
                    print("Available modules in 'bbox_detection' module: ")
                    print_module_names(legonet.bbox_detection)
                    #legonet.bbox_detection.load_state_dict(bbox_det_state_dict, strict=False)
                    load_submodule_weights(legonet.bbox_detection, bbox_det_state_dict,
                                           submodule_names=['backbone_1', 'find_1', 'where'], strict=False)

            if args.load_per_object_counting_weights and args.network_type == "both":
                per_object_state_dict = torch.load(args.per_object_weights_file, map_location=config.General.device)
                if args.estimate_type == 'withKeyPoints':
                    load_submodule_weights(legonet, per_object_state_dict,
                                       submodule_names = ['backbone_2', 'find_2', 'estimator'], strict=False)

            if args.load_per_object_attributes_weights and (args.network_type == "both_for_roots_2"\
                    or args.network_type == "both_Back2bFind2b"):
                per_object_state_dict = torch.load(args.per_object_weights_file, map_location=config.General.device)

                if args.network_type == "both_Back2bFind2b":
                    load_submodule_weights(legonet, per_object_state_dict,
                                           submodule_names = ['backbone_2', 'find_2',
                                                              'estimator_length', 'estimator_diameter', 'estimator_color',
                                                              'find_2_b', 'backbone_2_b'], strict=False)
                else:
                    load_submodule_weights(legonet, per_object_state_dict,
                                           submodule_names=['backbone_2', 'find_2',
                                                            'estimator_length', 'estimator_diameter', 'estimator_color'],
                                           strict=False)

        elif args.load_full_model_weights:
            model_state_dict = torch.load(args.full_model_weights, map_location=config.General.device)
            #legonet.load_state_dict(model_state_dict, strict=False)
            print("Available modules in the weights file:", list_checkpoint_modules(model_state_dict))
            print("Check keys:")
            if args.network_type == "counting_lean":
                load_submodule_weights(legonet, model_state_dict,
                                       submodule_names=['backbone', 'find', 'estimator'], strict=False)

            elif args.network_type == "counting_reg":
                load_submodule_weights(legonet, model_state_dict,
                                       submodule_names=['backbone', 'estimator'], strict=False)

            elif args.network_type == "both":
                if args.estimate_type == 'withKeyPoints':  # 'bbox_detection', "per_object_counting"
                    load_submodule_weights(legonet, model_state_dict,
                                           submodule_names=["bbox_detection", "per_object_counting"], strict=False)
            elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                if args.estimate_type == 'withKeyPoints':
                    load_submodule_weights(legonet, model_state_dict,
                                           submodule_names=["bbox_detection", "per_object_attributes"], strict=False)

            elif args.network_type == "bbox_detection":
                load_submodule_weights(legonet, model_state_dict,
                                       submodule_names=['backbone_1', 'find_1', 'where'], strict=False)

    elif args.save_from_model_file:
        # for initial saving of weights from old model pt files:

        if args.network_type == "both_Back2bFind2b":

            file_bbox = torch.load(args.model_path["bbox_path"], map_location=config.General.device)
            file_limit5Path = torch.load(args.model_path["limit5Path"], map_location=config.General.device)
            file_all_3setsPath = torch.load(args.model_path["all_3setsPath"], map_location=config.General.device)
            file_legonet = {"file_bbox": file_bbox,
                            "file_limit5Path": file_limit5Path,
                            "file_all_3setsPath": file_all_3setsPath
                            }
        else:
            file_legonet = torch.load(args.model_path, map_location=config.General.device)


        if not args.network_type == "both_Back2bFind2b":
            print("Available modules in model file:", list_checkpoint_modules(file_legonet.state_dict()))

        if args.network_type == 'bbox_detection':
            save_partial_weights(args, legonet, file_legonet, tasks=['bbox_detection'])

        elif args.network_type == "both":
            save_partial_weights(args, legonet, file_legonet, tasks=['bbox_detection', "per_object_counting"], output_name = args.output_name)

        elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
            save_partial_weights(args, legonet, file_legonet, tasks=['bbox_detection', "per_object_attributes"], output_name = args.output_name)

        elif args.network_type == "counting_lean" or args.network_type == "counting_reg":
            save_partial_weights(args, legonet, file_legonet, tasks=["per_image_attributes"], output_name = args.output_name)

    if (args.network_type == "bbox_detection" or args.network_type == "both" or args.network_type == "both_for_roots_2"
            or args.network_type == "both_Back2bFind2b"):
        if args.freeze_detection:
            legonet.freeze_detector()

    legonet = legonet.to(config.General.device)

    # if isinstance(legonet, torch.nn.DataParallel): # ToDo - check this
    #     legonet.module.freeze_bn()
    # else:
    #     legonet.freeze_bn()

    #freeze_bn(unwrap_model(legonet))

    if args.run_script=='Training':

        legonet.training = True

        optimizer = optim.Adam(legonet.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, eps=0.0001) #eps=0.0001 like in the keras code instead of the default verbose=True,
        loss_hist = collections.deque(maxlen=500)
        #legonet.train()

        print('Num training images: {}'.format(len(dataset_train)))

        best_rel_error = 100.0
        best_mAP = 0.0
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
                        epoch_loss_per_task = { #### change loss names!!
                            "classification": [],
                            "regression": [],
                            #"points": [],
                            "color": [],
                            "maps": [],
                            "length": [],
                            "diameter": []
                        }

                    elif  args.network_type == 'counting_lean':
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            #"counting": [],
                            "l1_estimation": [], #l1
                            "maps": []
                        }

                    elif args.network_type == 'both':
                        epoch_loss_per_task = {
                            "classification": [],
                            "regression": [],
                            "l1_counting": [],  # l1
                            "maps": []
                        }

                elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                    if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                        epoch_loss_per_task = {  #### change loss names!!
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
                            #counting_loss = l1_estimation_loss + maps_loss
                            #counting_loss = counting_loss.mean()
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
                        reg_estimation_loss = legonet([data['img'].to(config.General.device).float(), data['annot']]) #counting_loss
                        reg_estimation_loss = args.loss_weight * reg_estimation_loss
                        reg_estimation_loss = reg_estimation_loss.mean()
                        loss = reg_estimation_loss

                        reg_estimation_loss_value = reg_estimation_loss.item() #counting_loss_value

                    elif (args.network_type == 'both' or args.network_type == "both_for_roots_2"
                          or args.network_type == "both_Back2bFind2b"):

                        if torch.cuda.is_available():
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':

                                if args.network_type == 'both':
                                    bbox_classification_loss, bbox_regression_loss, l1_counting_loss, maps_loss = \
                                        legonet([data['img'].to(config.General.device).float(),
                                                 [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                                 torch.tensor(sampler.groups[iter_num])]) #, args.do_counting])

                                elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                    if 'points_annot' in data.keys():
                                        # for "both_for_roots_2", count loss is color loss
                                        bbox_classification_loss, bbox_regression_loss, color_loss, maps_loss, length_loss, diameter_loss=\
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                                     torch.tensor(sampler.groups[iter_num])]) #, args.do_counting])

                                    else:
                                        bbox_classification_loss, bbox_regression_loss, color_loss, maps_loss, length_loss, diameter_loss=\
                                            legonet([data['img'].to(config.General.device).float(),
                                                     [data['bbox_annot'].to(config.General.device), None],
                                                     torch.tensor(sampler.groups[iter_num])]) #, args.do_counting])

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                if args.network_type == 'both':
                                    bbox_classification_loss, bbox_regression_loss, reg_counting_loss =  legonet( #counting_loss
                                            [data['img'].to(config.General.device).float(),
                                             [data['bbox_annot'].to(config.General.device), data['points_annot']],
                                             torch.tensor(sampler.groups[iter_num]), True])

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
                            print("Iteration: " + iter_num+ " | CUDA not available")

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
                                #     counting_loss = l1 + maps_loss
                                #     counting_loss = counting_loss.mean()
                                    l1_counting_loss = l1_counting_loss.mean()
                                    maps_loss = maps_loss.mean()

                                    l1_counting_loss_value = l1_counting_loss.item()
                                    maps_loss_value = maps_loss.item()

                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss + l1_counting_loss + maps_loss  #counting_loss
                                    else:
                                        loss = l1_counting_loss + maps_loss

                                else:
                                    if bbox_detection_loss is not None:
                                        loss = bbox_detection_loss
                                    else:
                                        loss = None
                                    #counting_loss = None

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

                                    loss = bbox_detection_loss + reg_counting_loss
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
                                    #epoch_loss_per_task["counting"].append(counting_loss_value)
                                    epoch_loss_per_task["l1_estimation"].append(l1_estimation_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)

                            elif args.network_type == 'both':
                                if l1_counting_loss is not None and maps_loss is not None:
                                    #epoch_loss_per_task["counting"].append(counting_loss_value)
                                    epoch_loss_per_task["l1_counting"].append(l1_counting_loss_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)

                            elif args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and maps_loss is not None and length_loss is not None and diameter_loss is not None:
                                    #epoch_loss_per_task["attributes_estimation"].append(attibutes_estimation_loss_value)
                                    epoch_loss_per_task["color"].append(color_loss_value)
                                    epoch_loss_per_task["maps"].append(maps_loss_value)
                                    epoch_loss_per_task["length"].append(length_loss_value)
                                    epoch_loss_per_task["diameter"].append(diameter_loss_value)

                        elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':

                            if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                                if color_loss is not None and length_loss is not None and diameter_loss is not None:
                                    epoch_loss_per_task["color"].append(color_loss_value)
                                    epoch_loss_per_task["length"].append(length_loss_value)
                                    epoch_loss_per_task["diameter"].append(diameter_loss_value)

                            elif args.network_type == 'both':
                                if reg_counting_loss is not None:
                                    epoch_loss_per_task["reg_counting"].append(reg_counting_loss_value)

                            elif args.network_type == 'counting_reg':
                                if reg_estimation_loss is not None:
                                    epoch_loss_per_task["reg_estimation"].append(reg_estimation_loss_value)


                    # loss_hist.append(float(loss))
                    # epoch_loss.append(float(loss))

                    if args.network_type == 'bbox_detection': #args.dataset_type == 'coco':
                        print(
                            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(bbox_classification_loss), float(bbox_regression_loss),np.mean(loss_hist)))

                        del bbox_regression_loss
                        del bbox_classification_loss

                    elif args.network_type == 'counting_lean':   #args.dataset_type == 'csv_LCC':
                        print(
                            'Epoch: {} | Iteration: {} | l1 loss: {:1.5f} |  maps loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(l1_estimation_value), float(maps_loss_value), np.mean(loss_hist)))

                        #del counting_loss
                        del l1_estimation_loss
                        del maps_loss

                    elif args.network_type == 'counting_reg':   #args.dataset_type == 'csv_LCC':
                        print(
                            'Epoch: {} | Iteration: {} | estimation loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, float(reg_estimation_loss_value), np.mean(loss_hist)))

                        del reg_estimation_loss #counting_loss

                    if (args.network_type == 'both' or args.network_type == 'both_for_roots_2'
                            or args.network_type == "both_Back2bFind2b"):
                        if args.network_type == 'both':
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                                print(
                                     'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | KeyPointsMaps loss: {:1.5f} | '
                                     'Counting loss: {:1.5f}'.format(epoch_num, iter_num, detection_loss_value,
                                                                     maps_loss_value, l1_counting_loss_value))

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | Counting loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value, reg_counting_loss_value))

                        if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
                            if config.AttributeEstimation.estimate_type == 'withKeyPoints':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | KeyPointsMaps loss: {:1.5f} | '
                                    'Length loss: {:1.5f} | Diameter loss: {:1.5f} | Color loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value,
                                        maps_loss_value, length_loss_value, diameter_loss_value, color_loss_value))

                            elif config.AttributeEstimation.estimate_type == 'reg_fpn_p3_p7_min_sig':
                                print(
                                    'Epoch: {} | Iteration: {} | BBOX_detection loss: {:1.5f} | Length loss: {:1.5f} | '
                                    'Diameter loss: {:1.5f} | Color loss: {:1.5f}'.format(
                                        epoch_num, iter_num, detection_loss_value,
                                        length_loss_value, diameter_loss_value, color_loss_value))

                        # print(
                        #     'Epoch: {} | Iteration: {} | total loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        #         epoch_num, iter_num, float(total_loss), np.mean(loss_hist)))
                        #del total_loss
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


                except Exception as e:
                    print(e)
                    continue

            if args.network_type == 'bbox_detection':

                utils.printf('Epoch %d summary: classification mean %.5f, regression mean %.5f\n',
                        epoch_num,
                        np.mean(epoch_loss_per_task["classification"]),
                        np.mean(epoch_loss_per_task["regression"]))


                if args.dataset_type == "coco":

                    if args.eval_in_train:
                        print('Evaluating dataset')
                        coco_eval.evaluate_coco(dataset_val, legonet)

                elif args.dataset_type == 'kcsv' and args.kcsv_val is not None:

                    if args.eval_in_train:

                        legonet.eval()

                        utils.printf("Evaluating Dataset: ")
                        mAP, precision , recall = kcsv_eval_2.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val, legonet,
                                                                score_threshold=config.Detection.min_score,
                                                                iou_threshold=config.Detection.iou_threshold)
                        print(f'Current mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n')
                        with open(args.txt_results, 'a') as f:
                            f.write(f'Epoch: {epoch_num}, mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n')

                        if mAP > best_mAP:
                            best_mAP = mAP
                            print(f'Current best*: {best_mAP:.3f}\n')
                            remove_prevEpoch()
                            torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                          "legonet_epoch={}.pt".format(epoch_num)))
                        # else:
                        #     if epoch_num % 5 == 0:
                        #         torch.save(legonet.state_dict(), config.General.weights_dir + '/legonet_epoch={}.pt'.format(epoch_num))
                        #     utils.printf("\n")

                    elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        torch.save(legonet.state_dict(),
                                   os.path.join(config.General.weights_dir,"legonet_epoch={}.pt".format(epoch_num)))

                elif args.dataset_type ==  "roots_json" :

                    if args.evaluate_detection:
                        legonet.eval()
                        mAP, _, _ = kcsv_eval_2.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val, legonet,
                                                             score_threshold=config.Detection.min_score, iou_threshold=config.Detection.iou_threshold)
                        print(f'Current mAP = {best_mAP:.3f}\n')
                        # if len(precision)==0:
                        #     print('mAP: {:.3f} | precision: None | recall: None | prev_best_mAP: {:.3f} \n'.format(mAP, best_mAP))
                        # else:
                        #     print('mAP: {:.3f} | precision: {:.3f} | recall : {:.3f} | prev_best_mAP: {:.3f} \n'.format(mAP, precision[0], recall[0], best_mAP))

                        if mAP > best_mAP:
                            best_mAP = mAP
                            print(f'Current best*: {best_mAP:.3f}\n')
                            remove_prevEpoch()
                            torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                             'legonet_epoch={}.pt'.format(epoch_num)))

            elif args.network_type == 'counting_lean' or args.network_type == 'counting_reg':
            #     print('Evaluating dataset')
            #     count_agreement = counting_eval.eval(dataset_val, legonet, args)
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

                    rel_error = counting_eval.eval(dataloader_val, dataset_val, legonet, args)
                    #utils.printf("rel error: %.3f \n", rel_error)
                    print('Rel_error: {:.3f} | prev_best: {:.3f} \n'.format(rel_error, best_rel_error))

                    if rel_error < best_rel_error:
                        best_rel_error = rel_error
                        print(f'Current best*: {best_rel_error:.3f}\n')
                        remove_prevEpoch()
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir, 'legonet_epoch={}.pt'.format(epoch_num)))  # checkpoints\\

                elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                      'legonet_epoch={}.pt'.format(epoch_num)))#cont_

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
                    if args.network_type == "both_for_roots_2" or args.network_type == "both_Back2bFind2b":
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
                            np.mean(epoch_loss_per_task["reg_counting"]))

                print()

                if args.eval_in_train:

                    #assert args.evaluate_detection or args.do_counting
                    legonet.eval()

                    print()

                    if args.evaluate_detection:
                        print()
                        print("Object detection evaluation: ")

                        mAP, precision, recall = kcsv_eval_2.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val, legonet,
                                                             score_threshold=config.Detection.min_score, iou_threshold=config.Detection.iou_threshold)
                        #utils.printf("mAP = %.3f ", mAP)
                        if len(precision)==0:
                            print('mAP: {:.3f} | precision: None | recall: None | prev_best_mAP: {:.3f} \n'.format(mAP, best_mAP))
                        else:
                            #print('mAP: {:.3f} | precision: {:.3f} | recall : {:.3f} | prev_best_mAP: {:.3f} \n'.format(mAP, precision[0], recall[0], best_mAP))
                            print('mAP: {:.3f} | precision: {:.3f} | recall : {:.3f} | prev_best_mAP: {:.3f} \n'.format(
                                mAP, precision[-1], recall[-1], best_mAP))

                    utils.printf(args.network_type + " evaluation: ")

                    #if args.do_counting:
                    # utils.printf("Detection evaluation:\n")
                    # kcsv_eval_2.evaluateMAP(dataset_val, dataloader_val, sampler_val, legonet, score_threshold=0.05, show_PR_curve=False)
                    #kcsv_eval_2.evaluate(dataset_val, dataloader_val, sampler_val, legonet, score_threshold=[0.05, 0.5], iou_threshold=[0.5, 0.75], show_PR_curve=True)

                    #rel_error, _ = both_eval.eval(dataset_val, dataloader_val, sampler_val, legonet, to_draw=False, verbose=True, print_to_files=False)

                    if torch.cuda.is_available():
                        out = both_eval.eval(dataset_val, dataloader_val, sampler_val, legonet, to_draw=False, verbose=False,
                                       print_to_files=True, args=args)

                        if len(out) > 0:
                            rel_error = out[0]
                        else:
                            rel_error = -1

                        utils.printf("rel error: %.3f\n", rel_error)

                    else:
                        print("Iteration: " + iter_num + " | CUDA not available")

                    if rel_error < best_rel_error:
                        best_rel_error = rel_error
                        print(f'Current best*: {best_rel_error:.3f}\n')
                        remove_prevEpoch()
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                      'legonet_epoch={}.pt'.format(epoch_num)))

                elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                      'legonet_epoch={}.pt'.format(epoch_num)))#cont_

                if args.eval_in_train:
                    legonet.eval()
                    utils.printf("Start evaluation: ")

                    rel_error = counting_eval.eval(dataloader_val, dataset_val, legonet, args)
                    #utils.printf("rel error: %.3f \n", rel_error)
                    print('Rel_error: {:.3f} | prev_best: {:.3f} \n'.format(rel_error, best_rel_error))

                    if rel_error < best_rel_error:
                        best_rel_error = rel_error
                        print(f'Current best*: {best_rel_error:.3f}\n')
                        remove_prevEpoch()
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                      'legonet_epoch={}.pt'.format(epoch_num)))  # checkpoints\\

                        print(f'Current best*: {best_rel_error:.3f}\n')

                elif (epoch_num+1) % config.General.SAVE_EVERY_N_EPOCHS == 0:
                        torch.save(legonet.state_dict(), os.path.join(config.General.weights_dir,
                                                                      'legonet_epoch={}.pt'.format(epoch_num)))#cont_

            scheduler.step(np.mean(epoch_loss))

        legonet.eval()

    else: # Inference

        legonet.training = False
        legonet.eval()

        if config.General.NETWORK_TYPE == config.NetworkType.detection or \
                config.General.NETWORK_TYPE == config.NetworkType.detection_and_counting:
            # includes objects without points...
            if config.General.NETWORK_TYPE == config.NetworkType.detection or args.evaluate_detection:
                if args.evaluate_detection and args.have_GT:
                    print()
                    print("Object detection evaluation:\n")
                    if args.eval_detection_params:
                        average_precisions_all= kcsv_eval_2.evaluate(dataset_val, dataloader_val, sampler_val, legonet,
                                                                     iou_threshold=config.Detection.iou_threshold_list,
                                                                     score_threshold=config.Detection.min_score_list,
                                                                     save_path= args.test_dir, show_PR_curve=True)

                        print("iou, score, class_mAP")
                        for i in range(len(average_precisions_all)):
                            print(average_precisions_all[i])

                    else:
                        print(f"Results for min score: {config.Detection.min_score}, iou_threshold: {config.Detection.iou_threshold}")
                        mAP, precision, recall = kcsv_eval_2.evaluateMAP_simple(dataset_val, dataloader_val, sampler_val,
                                                                                legonet, score_threshold=config.Detection.min_score,
                                                                                iou_threshold=config.Detection.iou_threshold,
                                                                                generate_PR_curve=True)
                        print(f'mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}')

                        with open(args.txt_results, 'a') as f:
                            f.write(f'mAP = {mAP:.3f}, precision = {precision:.3f}, recall = {recall:.3f}\n')
                        if config.General.to_draw:
                            vis_bbox(dataloader_val, sampler_val, dataset_val, legonet, unnormalize = UnNormalizer())
                print()

            if args.evaluate_both:
                print("Attribute estimation evaluation:\n")

                out = both_eval.eval(dataset_val, dataloader_val, sampler_val, legonet, to_draw=config.General.to_draw,
                                     draw_maps = config.DrawProperties.DRAW_MAPS, verbose=True,
                                     draw_path = config.DrawProperties.save_img_path, print_to_files=True, args = args)

                if len(out)>0:
                    rel_error = out[0]
                else:
                    rel_error = -1

                utils.printf("rel error: %.3f \n", rel_error)

        else:
            utils.printf("Attribute estimation evaluation:\n")
            # # rel_error, _ = both_eval.eval(dataset_val, dataloader_val, sampler_val, legonet, to_draw=config.General.to_draw,
            # #                            verbose=True)
            # both_eval.eval(dataset_val, dataloader_val, sampler_val, legonet,
            #                               to_draw=config.General.to_draw,
            #                               verbose=True)

            if config.General.NETWORK_TYPE == config.NetworkType.counting_reg or config.General.NETWORK_TYPE == config.NetworkType.counting_lean\
                    or config.General.NETWORK_TYPE == config.NetworkType.counting_lean_multiple_out:

                rel_error = counting_eval.eval(dataloader_val, dataset_val, legonet, args)

                print("Final avg rel error:", rel_error)

            print('done')


