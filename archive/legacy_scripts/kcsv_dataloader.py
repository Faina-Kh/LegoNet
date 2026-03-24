from __future__ import print_function, division
import config

import sys
import os
import torch
import numpy as np
import random
import csv
import math

from six import raise_from

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image
import json




class KCSVDataset(Dataset):
    """KCSV dataset."""

    def __init__(self,
                 train_file,
                 class_list = None,
                 base_dir = None,
                 image_min_side = 800,
                 image_max_side = 1333,
                 pre_process = 'keras_like',
                 transform = None,
                 lean_version = "",
                 dataset_type = "",
                 have_GT = True
                 ):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        self.base_dir = base_dir
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.pre_process = pre_process
        self.lean_version = lean_version

        self.dataset_type = dataset_type

        self.have_GT = have_GT


        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(train_file)

        if self.dataset_type != "roots_json":
            # parse the provided class file
            try:
                with self._open_for_csv(self.class_list) as file:
                    reader = csv.reader(file, delimiter=',')
                    self.classes = self.load_classes(reader)
            except ValueError as e:
                raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key

            # csv with img_path, class_name, x1, y1, x2, y2,
            try:
                with self._open_for_csv(self.train_file) as file:
                    self.image_data_bbox, self.image_data_points_location = self._read_annotations(csv.reader(file, delimiter=','), self.classes)

            except ValueError as e:
                raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)

        else:
            result = {}
            result["root"] = 0
            self.classes = result

            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key

            if self.have_GT:
                self.json_data = json.load(open(self.train_file, 'r'))
                assert type(self.json_data) == dict, 'annotation file format {} not supported'.format(type(self.json_data))
            else:
                self.json_data = None

            if config.General.NETWORK_TYPE.name == "counting_lean_multiple_out":
                self.image_data_bbox, self.image_data_points_location, self.image_data_outputs = self._read_annotations(classes=self.classes, json_data=self.json_data)
            elif self.have_GT:
                self.image_data_bbox, self.image_data_points_location = self._read_annotations(classes=self.classes, json_data=self.json_data)

        self.img_info = self.get_img_info()


    def get_img_info(self):
        img_info = []
        if self.have_GT:
            image_names = list(self.image_data_bbox.keys())
        else:
            image_names = []
            list_files = os.listdir(self.base_dir)

            for im in list_files:
                if ".png" in im or ".jpg" in im:
                    image_names.append(im)

        self.image_ids = []
        id=0
        for name in image_names:
            current={}
            current['name'] = name
            current['path'] = os.path.join(self.base_dir, name)
            img = Image.open(current['path'])
            current['width'] = img.width
            current['height'] = img.height
            current['img_id'] = id
            self.image_ids.append(id)
            id += 1
            img_info.append(current)

            # find future scale
            orig_rows = current['height']
            orig_cols = current['width']
            smallest_side = min(orig_rows, orig_cols)
            scale = self.image_min_side / smallest_side
            largest_side = max(orig_rows, orig_cols)
            if largest_side * scale > self.image_max_side:
                scale = self.image_max_side / largest_side
            current['scale'] = scale

        return img_info

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx, pre_process = self.pre_process)

        if config.General.NETWORK_TYPE.name == "counting_lean_multiple_out":
            bbox_annot, points_annot = self.load_annotations(idx)

        else:
            if self.have_GT:
                bbox_annot, points_annot = self.load_annotations(idx)
            else:
                bbox_annot, points_annot = None, None

        if config.General.dataset_name =='grapes':
            boxes_to_remove = []
            for p_annot in points_annot[0]:
                if math.isnan(p_annot[1]):
                    boxes_to_remove.append(p_annot[2])

            points_counts=points_annot[0]
            points_coords = points_annot[1]
            for box_id in boxes_to_remove:
                #result1 = np.where(bbox_annot == box_id)
                #bbox_annot=np.delete(bbox_annot,result1[0][0],0)
                for idx1 in range(len(bbox_annot)):
                    if bbox_annot[idx1][5]==box_id:
                        bbox_annot = np.delete(bbox_annot,[idx1], 0)
                        break

                #points_counts = np.delete(points_counts, result2[0][0],0)
                for idx2 in range(len(points_counts)):
                    if points_counts[idx2][2]==box_id:
                        points_counts = np.delete(points_counts, [idx2], 0)
                        break


            points_annot=(points_counts,points_coords)

        if points_annot is None and bbox_annot is None:
            annot = {}

        elif points_annot is None and bbox_annot is not None:
            annot = {"bbox_annot": bbox_annot}

        else:
            annot = {"bbox_annot":bbox_annot, "points_annot":points_annot}

        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        if not config.General.NETWORK_TYPE == config.NetworkType.detection or config.Counting.double_counting:
            if self.have_GT:
                if len(sample['annot']['points_annot']) > 0:
                    annotations_group_num_of_points, annotations_group_points_center = sample['annot']['points_annot']
            else:
                annotations_group_num_of_points, annotations_group_points_center = [], []

        img = sample['img']

        if config.General.NETWORK_TYPE not in [config.NetworkType.detection, config.NetworkType.detection_and_counting]:
            # compute keypoints after the transformation are done
            if self.lean_version != "version_3":
                annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_1_R)
                annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_2_R)
                annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_3_R)
                annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_4_R)
                annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_5_R)
            else:
                annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_1_R, pyramid_level=3)
                annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_2_R, pyramid_level=4)
                annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_3_R, pyramid_level=5)
                annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_4_R, pyramid_level=6)
                annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                             radius=config.Counting.map_5_R, pyramid_level=7)

            sample['annot']['points_annot'] = [annotations_group_num_of_points, annotation_map_1, annotation_map_2, annotation_map_3,
                               annotation_map_4, annotation_map_5]

            sample['lean_version'] = self.lean_version

        sample['img_name'] = self.img_info[self.image_ids[idx]]

        if config.detect_with_points.detect_points and len(points_annot) >0:
            points_by_box = {}
            for i in range(len(points_annot[1])):

                id = str(int(points_annot[1][i][3]))
                if id in points_by_box.keys():
                    points_by_box[id].append(points_annot[1][i][:2])
                else:
                    points_by_box[id] = []
                    points_by_box[id].append(points_annot[1][i][:2])

            per_obj_maps = []
            for id in points_by_box.keys():
                box_points = points_by_box[id]
                # reshape to [N, (x,y)]
                points_num = len(box_points)
                box_points = np.array([np.array(box_points).reshape(points_num, 2)])

                annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, box_points,
                                                                             radius=config.Counting.map_1_R)
                annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, box_points,
                                                                             radius=config.Counting.map_2_R)
                annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, box_points,
                                                                             radius=config.Counting.map_3_R)
                annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, box_points,
                                                                             radius=config.Counting.map_4_R)
                annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, box_points,
                                                                             radius=config.Counting.map_5_R)

                per_obj_maps.append([float(id), [annotation_map_1, annotation_map_2, annotation_map_3, annotation_map_4,
                                                    annotation_map_5]])


            sample['annot']['per_obj_maps'] = per_obj_maps

        return sample

    def load_image(self, image_index, pre_process):
        image_path = os.path.join(self.base_dir, self.img_info[self.image_ids[image_index]]['name'])
        image = np.asarray(Image.open(image_path).convert('RGB'))

        if pre_process == "keras_like":
            # transform the image to bgr
            return image[:, :, ::-1].copy()

        else:
            return image

    def get_output_counting(self, image_name):
        annotations_group_num_of_points = self.load_annotations_num_of_points(image_name) #[count,class,box_id,length,diameter]
        annotations_group_points_center = self.load_annotations_points_centers(image_name) #[x,y,box_id]

        # # check validity of annotations
        # image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves = self.filter_annotations(
        #     image_group, annotations_group_leaves_center, annotations_group_num_of_leaves, group)

        return annotations_group_num_of_points, annotations_group_points_center

    def load_annotations_points_centers(self, image_name):
        annots = self.image_data_points_location[image_name]
        centers = np.zeros((len(annots), 4))

        for idx, annot in enumerate(annots):
            class_name = annot['points_class']
            centers[idx, 0] = float(annot['x'])
            centers[idx, 1] = float(annot['y'])
            centers[idx, 2] = self.name_to_label(class_name)
            if 'bbox_id' in annot.keys():
                centers[idx, 3] = annot['bbox_id']
            else:
                centers[idx, 3] = -1


        return centers

    def load_annotations_num_of_points(self, image_name):
        #ToDo add support in multi class of points
        annots = self.image_data_bbox[image_name]
        counts = np.zeros((len(annots), 5)) # counts per bbox

        if len(annots) > 0: # if there are bboxes in the image
            for idx, annot in enumerate(annots):
                if len(annot['points'])>0:
                    class_name = annot['points_class']
                    if config.detect_and_count.type != 'both_for_roots_2':
                        counts[idx, 0] = float(annot['points_count'])
                    else:
                        counts[idx, 0] = float(annot['Root_Color'])

                    counts[idx, 1] = self.name_to_label(class_name)
                    counts[idx, 2] = annot['bbox_id']

                    if self.labels[0]=="root":
                        counts[idx, 3] = annot['Root_Length']
                        counts[idx, 4] = annot['Root_Diameter']


                else:
                    counts[idx, 0]=0
                    counts[idx, 1] = None
                    counts[idx, 2] = annot['bbox_id']

                    if self.labels[0]=="root":
                        counts[idx, 3] = None
                        counts[idx, 4] = None

        else:
            annots = self.image_data_points_location[image_name]# no bboxes, only points
            counts[0, 0] = len(annots)
            counts[0, 1] = None
            counts[0, 2] = -1
            if self.labels[0] == "root":
                counts[0, 3] = None
                counts[0, 4] = None

        return counts

    def load_annotations(self, image_index):

        # get ground truth annotations
        image_name = self.img_info[self.image_ids[image_index]]['name']
        #print(image_name)
        if config.General.NETWORK_TYPE == config.NetworkType.detection or config.General.NETWORK_TYPE == config.NetworkType.detection_and_counting:
            annotation_list = self.image_data_bbox[image_name]

            ###################################################################################
            # bbox annotations
            ###################################################################################
            bbox_annotations = np.zeros((0, 6))

            # some images appear to miss annotations (like image with id 257034)
            if len(annotation_list) == 0:
                return bbox_annotations, []

            to_remove=[]
            per_obj_maps = []
            # parse annotations
            for idx, a in enumerate(annotation_list):
                # some annotations have basically no width / height, skip them
                x1 = a['x1']
                x2 = a['x2']
                y1 = a['y1']
                y2 = a['y2']

                if (x2 - x1) < 1 or (y2 - y1) < 1:

                    if (x1 - x2) >= 1 and not (y2 - y1) < 1:
                        self.image_data_bbox[image_name][idx]['x1'] = x2
                        self.image_data_bbox[image_name][idx]['x2'] = x1
                        x1 = self.image_data_bbox[image_name][idx]['x1']
                        x2 = self.image_data_bbox[image_name][idx]['x2']

                    else:
                        to_remove.append(a)
                        continue

                annotation = np.zeros((1, 6))

                annotation[0, 0] = x1
                annotation[0, 1] = y1
                annotation[0, 2] = x2
                annotation[0, 3] = y2

                annotation[0, 4] = self.name_to_label(a['bbox_class'])

                annotation[0, 5] = a['bbox_id']

                bbox_annotations = np.append(bbox_annotations, annotation, axis=0)

            if len(to_remove)>0:
                for a in to_remove:
                    self.image_data_bbox[image_name].remove(a)
        ###################################################################################
        # points annotations
        ###################################################################################

        if config.General.NETWORK_TYPE == config.NetworkType.detection and not config.Counting.double_counting:
            points_annotations = None
        else:
            points_annotations = self.get_output_counting(image_name) #[counts: count, point class, box id, centers: x,y, point class, box id]

        return bbox_annotations, points_annotations



    def _read_annotations(self, csv_reader=None, classes=None, json_data=None):
        result_bbox = {}
        result_points = {}
        bbox_id = 0

        if json_data is not None:

            image_outputs = {}

            points_class = self.labels[0]

            for key in json_data.keys():
                current = json_data[key]
                if 'original_name' in current.keys():
                    img_file = current['original_name']
                else:
                    img_file = current['processed_name']

                #if img_file == 'RAMATNEGEVWINES_T025_L077_2012.08.29_091305_001_YSI .jpg':
                #    a=1


                img = skimage.io.imread(os.path.join(self.base_dir,img_file))
                img_H = img.shape[0]
                img_W = img.shape[1]

                if img_file not in result_bbox:
                    result_bbox[img_file] = []
                if img_file not in result_points:
                    result_points[img_file] = []

                bbox_id = 0
                for img_key in current.keys():
                    if "root_" in img_key:
                        bbox_id +=1
                        #bbox_id = int(img_key.split("_")[1])
                        points_in_box = current[img_key]['points']
                        points_num = int(len(points_in_box) / 2)

                        min_x=1000000; min_y=1000000; max_x=0; max_y=0

                        # result_points
                        j = 0
                        for i in range(points_num):
                            x = points_in_box[j]
                            y = points_in_box[j + 1]
                            result_points[img_file].append({'x': x, 'y': y, 'points_class': points_class, 'bbox_id': bbox_id})
                            j = j + 2

                            if x < min_x:
                                min_x = x

                            if y < min_y:
                                min_y = y

                            if x > max_x:
                                max_x = x

                            if y > max_y:
                                max_y = y


                        # enlarge the box
                        x1 = np.maximum(min_x - 10, 0)
                        y1 = np.maximum(min_y - 10, 0)
                        x2 = np.minimum(max_x + 10, img_W)
                        y2 = np.minimum(max_y + 10, img_H)


                        # result_bbox

                        # find box coordinates
                        if config.detect_and_count.type != 'both_for_roots_2':
                            result_bbox[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                                          'bbox_class': points_class, 'bbox_id': bbox_id, 'points': points_in_box,
                                                          'points_count': points_num,
                                                          'points_class': points_class,
                                                          'Root_Length': current[img_key]['Root_Length'],
                                                          'Root_Diameter':current[img_key]['Root_Diameter']})

                        else:
                            if 'Root_Color' in current[img_key].keys():
                                if current[img_key]['Root_Color'] == 'White':
                                    color = 1
                                else:
                                    color =0
                            else:
                                color = -1


                            result_bbox[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                                          'bbox_class': points_class, 'bbox_id': bbox_id,
                                                          'points': points_in_box,
                                                          'points_count': points_num,
                                                          'points_class': points_class,
                                                          'Root_Length': current[img_key]['Root_Length'],
                                                          'Root_Diameter': current[img_key]['Root_Diameter'],
                                                          'Root_Color': color})


                    if config.General.NETWORK_TYPE.name == "counting_lean_multiple_out":
                        image_outputs[img_file] = {}
                        image_outputs[img_file]["TRL"] = current['TRL']
                        image_outputs[img_file]['roots_num'] = current['roots_num']
                        image_outputs[img_file]['roots_dia_mean'] = current['roots_dia_mean']
                        image_outputs[img_file]['roots_dia_std'] = current['roots_dia_std']

        else:
            for line, row in enumerate(csv_reader):
                line += 1

                if len(row) == 6:
                    try:
                        img_file, class_name, x1, y1, x2, y2 = row[:6]

                        if x2 == "":
                            img_file, class_name, x1, y1 = row[:4]
                            x2, y2 = "", ""

                    except ValueError:
                        raise_from(ValueError(
                            'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,x1,y1,class_name,,,,\''.format(line)),
                                   None)

                elif len(row) == 4:
                    try:
                        img_file, class_name, x1, y1 = row[:4]
                        x2, y2 = "", ""

                    except ValueError:
                        raise_from(ValueError(
                            'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,x1,y1,class_name,,,,\''.format(line)),
                                   None)


                # if x2 !=  ""  and y2 != "": # it's a bbox
                #     if img_file not in result_bbox:
                #         result_bbox[img_file] = []
                #
                # else: # it's a point
                #     if img_file not in result_points:
                #         result_points[img_file] = []
                if img_file not in result_bbox:
                    result_bbox[img_file] = []

                if img_file not in result_points:
                     result_points[img_file] = []

                # If a row contains only an image path, it's an image without annotations.
                if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                    continue

                # check if the current class name is correctly present
                if class_name not in classes:
                    raise ValueError(
                        'line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

                x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
                y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))

                if x2!= "": #it's a bbox
                    x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
                    y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

                    # Check that the bounding box is valid.
                    if x2 <= x1:
                        raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                    if y2 <= y1:
                        raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                    result_bbox[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                                  'bbox_class': class_name, 'bbox_id': bbox_id, 'points':[] })
                    bbox_id += 1

                else:
                    result_points[img_file].append({'x': x1, 'y': y1, 'points_class': class_name})

            if config.General.NETWORK_TYPE == config.NetworkType.detection_and_counting or config.Counting.double_counting:
                result_bbox = self.points_to_bbox(result_bbox, result_points)

        if config.General.NETWORK_TYPE.name == "counting_lean_multiple_out":
            return result_bbox, result_points, image_outputs

        return result_bbox, result_points

    def points_to_bbox(self, result_bbox, result_points):

        for img in result_points.keys():
            for i in range(len(result_points[img])):
                x1_p,y1_p = result_points[img][i]['x'], result_points[img][i]['y']

                for j in range(len(result_bbox[img])):
                    x1,y1,x2,y2 = result_bbox[img][j]['x1'], result_bbox[img][j]['y1'],result_bbox[img][j]['x2'],result_bbox[img][j]['y2']
                    if x1_p >= x1 and x1_p <= x2 and y1_p >= y1 and y1_p <= y2:
                        result_points[img][i]['bbox_id'] = result_bbox[img][j]['bbox_id']
                        result_bbox[img][j]['points'].append(result_points[img][i])
                        break

        for img in result_bbox.keys():
            for j in range(len(result_bbox[img])):
                if len(result_bbox[img][j]['points'])>0:
                    result_bbox[img][j]['points_count'] = len(result_bbox[img][j]['points'])
                    result_bbox[img][j]['points_class'] = result_points[img][0]['points_class']


        return result_bbox

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image_info = self.img_info[self.image_ids[image_index]]
        return float(image_info['width']) / float(image_info['height'])


    def image_output_shape(self, image_shape, pyramid_level=3):
        return (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)


    def images_ratios(self, image_shape, output_shape):
        return output_shape / np.array(image_shape[:2])


    def create_gausian_mask(self, center_point, nCols, nRows, q=99, radius=(5, 5)):
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

        x2 = (((x - round(center_point[0])) * s) / radius[0]) ** 2
        y2 = (((y - round(center_point[1])) * s) / radius[1]) ** 2

        p = np.exp(-0.5 * (x2 + y2))

        p[np.where(p < np.percentile(p, q))] = 0

        p = p / np.max(p)
        if not np.isfinite(p).all() or not np.isfinite(p).all():
            print('divide by zero')
        return p


    def compute_keypoints_targets_multi_maps(self, image_shape, annotations_points_centers_a, radius=(5, 5), pyramid_level=3):
        # resize transformed-image and annotations
        import copy
        annotations_points_centers = copy.deepcopy(annotations_points_centers_a)
        # here we should resize image too and then check it with the annotations
        output_shape = self.image_output_shape(image_shape, pyramid_level=pyramid_level)
        image_ratio = self.images_ratios(image_shape, output_shape)

        if len(annotations_points_centers) == 0:
            return [np.zeros(output_shape)]

        per_img_anns = []
        img_num=len(annotations_points_centers)
        for i in range(img_num):
            current_points = annotations_points_centers[i] #[N, [x, y, points class, bbox_id]]
            current_points[:, :2] = current_points[:, :2] * image_ratio

            annotations = np.zeros(output_shape)
            for j in range(current_points.shape[0]):
                gaussian_map = self.create_gausian_mask(current_points[j, :2], output_shape[1], output_shape[0],
                                                   radius=radius)
                # each center point in the GT will be 1 in the annotation map
                annotations = np.maximum(annotations, gaussian_map)

            if np.isnan(annotations).any():
                raise ("nan was found")

            per_img_anns.append(annotations)

        return per_img_anns

















