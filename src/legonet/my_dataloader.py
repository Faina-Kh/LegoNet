from __future__ import print_function, division
import sys
import os
from pathlib import Path
import torch
import numpy as np
import random
import csv
from legonet import config
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

import time


########################################################################################################################
# help functions

########################################################################################################################


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, #'images',
                                   self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return max(self.classes.values())+1 #80


class csv_LCCDataset(Dataset):


    def __init__(
            self,
            csv_leaf_number_file,
            csv_leaf_location_file,
            base_dir=None,
            image_min_side=800,
            image_max_side=1333,
            pre_process = 'keras_like',
            ann_type = None,
            transform = None,
            json_file = None,
            have_GT = True

    ):
        self.csv_leaf_number_file = csv_leaf_number_file
        self.csv_leaf_location_file = csv_leaf_location_file

        self.base_dir = base_dir
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.pre_process = pre_process
        self.ann_type = ann_type
        self.transform = transform
        self.json_file = json_file

        self.have_GT = have_GT

        self.bgr_images_names = []
        self.centers_images_names = []
        self.image_data_leaf_number = {}
        self.image_data_leaf_location = {}

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_leaf_number_file)


        self.labels = {'0': 'leaves'}
        self.classes = {'leaves': 0}


        if self.have_GT and csv_leaf_number_file:
            # csv with img_path, num_of_leaves
        #try: # for leaf data- requiered int anns
            with self._open_for_csv(csv_leaf_number_file) as file:
                self.image_data_leaf_number = self._read_annotations_NOL(csv.reader(file, delimiter=','))
        #except ValueError as e:
        #    raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_leaf_number_file, e)), None)
            rgb_images_names = list(self.image_data_leaf_number.keys())
            self.bgr_images_names = rgb_images_names

            # csv with img_path, x, y
            if csv_leaf_location_file != "":
                try:
                    with self._open_for_csv(csv_leaf_location_file) as file:
                        self.image_data_leaf_location = self._read_annotations_leaves_locations(csv.reader(file, delimiter=','))
                except ValueError as e:
                    raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_leaf_location_file, e)), None)
                self.centers_images_names = [x.replace('rgb', 'centers') for x in rgb_images_names]
                if config.General.dataset_name!= 'roots': #'root' not in self.base_dir and 'Root' not in self.base_dir:
                    assert set(list(self.image_data_leaf_location.keys())) == set(
                        self.centers_images_names), 'there are some missing centers annotations'


            if json_file !=None:
                self.json_data = json.load(open(json_file, 'r'))
                assert type(self.json_data) == dict, 'annotation file format {} not supported'.format(type(self.json_data))



        else:
            rgb_images_names = os.listdir(self.base_dir)
            rgb_images_names_a = []

            for im in rgb_images_names:
                if 'CVPPP' in base_dir:
                    if im.lower().endswith('_rgb.png'):
                        rgb_images_names_a.append(im)
                else:
                    if Path(im).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        rgb_images_names_a.append(im)
            self.bgr_images_names = rgb_images_names_a


    def image_path_rgb(self, image_index):
        #print(os.path.join(self.base_dir, self.bgr_images_names[image_index]))
        return os.path.join(self.base_dir, self.bgr_images_names[image_index])

    def num_classes(self):
        return max(self.classes.values()) + 1


    def __len__(self):#size(self):
        return len(self.bgr_images_names)


    def load_image(self, image_index, pre_process):
        #print(self.image_path_rgb(image_index).split("\\")[-1])
        image = np.asarray(Image.open(self.image_path_rgb(image_index)).convert('RGB'))

        if pre_process == "keras_like":
            # transform the image to bgr
            return image[:, :, ::-1].copy()

        else:
            return image


    def get_output_forV20(self, group):
        annotations_group_num_of_leaves = self.load_annotations_group_num_of_leaves(group)
        if len(self.image_data_leaf_location)>0:
            annotations_group_leaves_center = self.load_annotations_group_leaves_center(group)

            # # check validity of annotations
            # image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves = self.filter_annotations(
            #     image_group, annotations_group_leaves_center, annotations_group_num_of_leaves, group)
            return annotations_group_num_of_leaves[0], annotations_group_leaves_center[0][0]
        else:
            return annotations_group_num_of_leaves[0]

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

        #p[np.where(p < np.percentile(p, q))] = 0
        p[np.where(p <= np.percentile(p, q))] = 0

        if np.max(p)>0:
            p = p / np.max(p)

        if not np.isfinite(p).all() or not np.isfinite(p).all():
            print('divide by zero')
        return p


    def compute_keypoints_targets_multi_maps(self, image_shape, annotations_leaves_centers_a, radius=(5, 5), pyramid_level=3):
        # resize transformed-image and annotations
        import copy
        annotations_leaves_centers = copy.deepcopy(annotations_leaves_centers_a)
        # here we should resize image too and then check it with the annotations
        output_shape = self.image_output_shape(image_shape, pyramid_level=pyramid_level)
        image_ratio = self.images_ratios(image_shape, output_shape)
        annotations_leaves_centers[:, :2] = annotations_leaves_centers[:, :2] * image_ratio
        annotations = np.zeros(output_shape)

        for i in range(annotations_leaves_centers.shape[0]):
            if np.all(annotations_leaves_centers_a==[0,0,0]): #np.sum(annotations_leaves_centers)==0   # there are no gt points
                continue

            #time.sleep(0.01)
            gaussian_map = self.create_gausian_mask(annotations_leaves_centers[i, :2], output_shape[1], output_shape[0],
                                               radius=radius)

            #draw_path = "D:\\Faina\\roots_project\\Rootfly_cam3\\Results\\val_points_for_roots_eval\\points_pred_img"
            #plt.imsave(draw_path + '/' + "aaa" + '_map_'+str(i)+ '_anno.png', annotations)

            #time.sleep(0.01)
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)

        #time.sleep(0.01)
        if np.isnan(annotations).any():
            raise ("nan was found")
        return annotations


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


    def _read_annotations_NOL(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, num_of_leaves = row[:2]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file, num_of_leaves\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (num_of_leaves) == (''):
                raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

            # Check that the bounding box is valid.
            if config.General.dataset_name!= 'roots': #'root' not in self.base_dir and 'Root' not in self.base_dir:
                if int(float(num_of_leaves)) <= 0:
                    raise ValueError('num_of_leaves must be higher than 0 but is {}'.format(num_of_leaves))

            result[img_file].append({'num_of_leaves': num_of_leaves, 'class': 'leaves'})
        return result

    #
    # def _read_annotations_leaves_locations(self, csv_reader):
    #     result = {}
    #     for line, row in enumerate(csv_reader):
    #         line += 1
    #
    #         try:
    #             img_file, x, y = row[:3]
    #         except ValueError:
    #             raise_from(ValueError('line {}: format should be \'img_file,x,y\' or \'img_file,,,,,\''.format(line)),
    #                        None)
    #
    #         if img_file not in result:
    #             result[img_file] = []
    #
    #         # If a row contains only an image path, it's an image without annotations.
    #         if (x, y) == ('', ''):
    #             raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)
    #
    #         x1 = self._parse(x, int, 'line {}: malformed x1: {{}}'.format(line))
    #         y1 = self._parse(y, int, 'line {}: malformed y1: {{}}'.format(line))
    #
    #         # Check that the bounding box is valid.
    #         if x1 < 0:
    #             raise ValueError('line {}: x ({}) must be higher than 0 ({})'.format(line, x))
    #         if y1 < 0:
    #             raise ValueError('line {}: y ({}) must be higher than 0 ({})'.format(line, y))
    #
    #         result[img_file].append({'x': x, 'y': y, 'class': 'leaves'})
    #     return result
    #

    def _read_annotations_leaves_locations(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            if config.General.dataset_name!= 'roots': #"root" not in self.base_dir and "Root" not in self.base_dir:
                try:
                    img_file, x, y = row[:3]
                except ValueError:
                    raise_from(ValueError('line {}: format should be \'img_file,x,y\' or \'img_file,,,,,\''.format(line)),
                               None)

                if img_file not in result:
                    result[img_file] = []

                # If a row contains only an image path, it's an image without annotations.
                if (x, y) == ('', ''):
                    raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

                x1 = self._parse(x, int, 'line {}: malformed x1: {{}}'.format(line))
                y1 = self._parse(y, int, 'line {}: malformed y1: {{}}'.format(line))

                # Check that the bounding box is valid.
                if x1 < 0:
                    raise ValueError('line {}: x ({}) must be higher than 0 ({})'.format(line, x))
                if y1 < 0:
                    raise ValueError('line {}: y ({}) must be higher than 0 ({})'.format(line, y))

                result[img_file].append({'x': x, 'y': y, 'class': 'leaves'})

            else:
                img_file = row[0]

                # img = skimage.io.imread(os.path.join(self.base_dir, img_file))
                # img_w = img.shape[1]
                # img_h = img.shape[0]

                if img_file not in result:
                    result[img_file] = []

                if len(row)==1:
                    result[img_file].append({})
                    continue

                if row[1]=="":
                    continue

                points_in_row = (len(row)-1)/2
                count=0
                i=1
                while count < points_in_row:
                    count+=1

                    # # If a row contains only an image path, it's an image without annotations.
                    # if (x, y) == ('', ''):
                    #     raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

                    x, y = row[i], row[i+1]
                    x1 = self._parse(x, int, 'line {}: malformed x1: {{}}'.format(line))
                    y1 = self._parse(y, int, 'line {}: malformed y1: {{}}'.format(line))

                    # Check that the bounding box is valid.
                    if x1 < 0:
                        # raise ValueError('line {}: x ({}) must be higher than 0 ({})'.format(line, x))
                        x = "0"

                    # if x1 > (img_w-1):
                    #     x = str(img_w-1)

                    if y1 < 0:
                        #raise ValueError('line {}: y ({}) must be higher than 0 ({})'.format(line, y))
                        y = "0"

                    # if y1 > (img_h-1):
                    #     y = str(img_h-1)

                    result[img_file].append({'x': x, 'y': y, 'class': 'leaves'})

                    i+=2

        # import cv2
        # image = cv2.imread(os.path.join(self.base_dir, img_file))
        #
        # points= [(227,336), (283,341), (313,352)]
        # H_ratio = 1944 / 480
        # W_ratio = 2592 / 640
        # for p in points:
        #     print(p)
        #     image = cv2.circle(image,   (int(p[0] * W_ratio), int(p[1] * H_ratio)), radius=5, color=(255, 0, 0), thickness=2)
        #
        # cv2.imwrite(os.path.join(self.base_dir,"points_"+img_file),image)

        return result


    def load_annotations_group_leaves_center(self, group):
        return [[self.load_annotations_leaves_centers(image_index) for image_index in group]]


    def load_annotations_leaves_centers(self, image_index):
        path = self.centers_images_names[image_index]
        annots = self.image_data_leaf_location[path]
        centers = np.zeros((len(annots), 3))
        if len(annots[0]) > 0:
            for idx, annot in enumerate(annots):
                class_name = annot['class']
                centers[idx, 0] = float(annot['x'])
                centers[idx, 1] = float(annot['y'])
                centers[idx, 2] = self.name_to_label(class_name)

        # else:
        #     a=1

        return centers


    def load_annotations_group_num_of_leaves(self, group):
        return [self.load_annotations_num_of_leaves(image_index) for image_index in group]


    def load_annotations_num_of_leaves(self, image_index):
        path = self.bgr_images_names[image_index]
        # if path == "T032_L116_2013.09.09_211622_006.jpg":
        #     a=1
        annots = self.image_data_leaf_number[path]
        counts = np.zeros((len(annots), 2))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            counts[idx, 0] = float(annot['num_of_leaves'])
            counts[idx, 1] = self.name_to_label(class_name)

        return counts


    def name_to_label(self, name):
        return self.classes[name]


    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_path_rgb(image_index))
        return float(image.width) / float(image.height)


    def __getitem__(self, idx):

        img = self.load_image(image_index = idx, pre_process = self.pre_process)

        # if self.bgr_images_names[idx] == 'MOPMELON_T008_L018_2019.01.28_123012_009_JEE.jpg':
        #     a = 1

        if self.have_GT:
            annot = self.get_output_forV20([idx])

        if self.have_GT:
            sample = {'img': img, 'annot': annot}

        else:
            sample = {'img': img}

        if self.transform:
            sample = self.transform(sample)

        img = sample['img']

        if self.have_GT:
            if len(sample['annot']) == 1:
                annotations_group_num_of_leaves = sample['annot']
                sample['annot'] = [annotations_group_num_of_leaves[0]]

            elif len(sample['annot']) == 2:
                annotations_group_num_of_leaves, annotations_group_leaves_center = sample['annot']

                # compute keypoints after the transformation are done
                annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_leaves_center, radius=config.AttributeEstimation.map_1_R)
                annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_leaves_center, radius=config.AttributeEstimation.map_2_R)
                annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_leaves_center, radius=config.AttributeEstimation.map_3_R)
                annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_leaves_center, radius=config.AttributeEstimation.map_4_R)
                annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_leaves_center, radius=config.AttributeEstimation.map_5_R)


                sample['annot'] = [annotations_group_num_of_leaves[0], annotation_map_1, annotation_map_2, annotation_map_3, annotation_map_4, annotation_map_5]
            # plt.imsave(draw_path + '/' + 'ann_map_'+str(1)+ '_anno.png', annotation_map_1)


        return sample


class KCSVDataset(Dataset):
    """KCSV dataset."""

    def __init__(self,
                 input_file,
                 class_list = None,
                 base_dir = None,
                 image_min_side = 800,
                 image_max_side = 1333,
                 pre_process = 'keras_like',
                 transform = None,
                 dataset_type = "",
                 have_GT = True
                 ):
        """
        Args:
            input_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = input_file
        self.class_list = class_list
        self.transform = transform

        self.base_dir = base_dir
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.pre_process = pre_process

        self.dataset_type = dataset_type

        self.have_GT = have_GT

        self.is_roots_2 = config.Detect_and_Estimate.type == "per_object_attributes" or config.Detect_and_Estimate.type =="per_object_attributes_multibranch"

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(input_file)

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

            if self.have_GT:
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

            if self.have_GT:
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
                if Path(im).suffix.lower() in {".jpg", ".jpeg", ".png"}:
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

        if self.have_GT:
            bbox_annot, points_annot = self.load_annotations(idx)
        else:
            bbox_annot, points_annot = None, None

        if points_annot is None and bbox_annot is None:
            annot = {}

        elif points_annot is None and bbox_annot is not None:
            annot = {"bbox_annot": bbox_annot}

        else:

            # if config.General.dataset_name == 'grapes': # and filter_empty...
                # boxes_to_remove = []
                # for p_annot in points_annot[0]:
                #     if math.isnan(p_annot[1]):
                #         boxes_to_remove.append(p_annot[2])

                # points_counts = points_annot[0]
                # points_coords = points_annot[1]
                # for box_id in boxes_to_remove:
                #     # result1 = np.where(bbox_annot == box_id)
                #     # bbox_annot=np.delete(bbox_annot,result1[0][0],0)
                #     for idx1 in range(len(bbox_annot)):
                #         if bbox_annot[idx1][5] == box_id:
                #             bbox_annot = np.delete(bbox_annot, [idx1], 0)
                #             break
                #
                #     # points_counts = np.delete(points_counts, result2[0][0],0)
                #     for idx2 in range(len(points_counts)):
                #         if points_counts[idx2][2] == box_id:
                #             points_counts = np.delete(points_counts, [idx2], 0)
                #             break
                #
                # points_annot = (points_counts, points_coords)

            annot = {"bbox_annot":bbox_annot, "points_annot":points_annot}

        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        if not config.General.NETWORK_TYPE == config.NetworkType.detection:
            if self.have_GT:
                if len(sample['annot']['points_annot']) > 0:
                    annotations_group_num_of_points, annotations_group_points_center = sample['annot']['points_annot']
            else:
                annotations_group_num_of_points, annotations_group_points_center = [], []

        img = sample['img']

        if config.General.NETWORK_TYPE not in [config.NetworkType.detection, config.NetworkType.detection_and_estimation]:
            # compute keypoints after the transformation are done
            #if self.lean_version != "version_3":
            annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                         radius=config.AttributeEstimation.map_1_R)
            annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                         radius=config.AttributeEstimation.map_2_R)
            annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                         radius=config.AttributeEstimation.map_3_R)
            annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                         radius=config.AttributeEstimation.map_4_R)
            annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
                                                                         radius=config.AttributeEstimation.map_5_R)
            # else:
            #     annotation_map_1 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
            #                                                                  radius=config.Counting.map_1_R, pyramid_level=3)
            #     annotation_map_2 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
            #                                                                  radius=config.Counting.map_2_R, pyramid_level=4)
            #     annotation_map_3 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
            #                                                                  radius=config.Counting.map_3_R, pyramid_level=5)
            #     annotation_map_4 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
            #                                                                  radius=config.Counting.map_4_R, pyramid_level=6)
            #     annotation_map_5 = self.compute_keypoints_targets_multi_maps(img.shape, annotations_group_points_center,
            #                                                                  radius=config.Counting.map_5_R, pyramid_level=7)

            sample['annot']['points_annot'] = [annotations_group_num_of_points, annotation_map_1, annotation_map_2, annotation_map_3,
                               annotation_map_4, annotation_map_5]

            #sample['lean_version'] = self.lean_version

        sample['img_name'] = self.img_info[self.image_ids[idx]]

        return sample

    def load_image(self, image_index, pre_process):
        image_path = os.path.join(self.base_dir, self.img_info[self.image_ids[image_index]]['name'])
        image = np.asarray(Image.open(image_path).convert('RGB'))
        if pre_process == "keras_like":
            # transform the image to bgr
            return image[:, :, ::-1].copy()
        else:
            return image
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
                    if not self.is_roots_2:
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
        if config.General.NETWORK_TYPE == config.NetworkType.detection or config.General.NETWORK_TYPE == config.NetworkType.detection_and_estimation:
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

        if config.General.NETWORK_TYPE == config.NetworkType.detection and not config.General.filter_empty_bbox:
            points_annotations = None
        else:
            points_annotations = self.get_output_counting(image_name) #[counts: count, point class, box id, centers: x,y, point class, box id]

        # if config.General.dataset_name == 'grapes':  # and filter_empty...
        #     boxes_to_remove = []
        #     for p_annot in points_annotations[0]:
        #         if math.isnan(p_annot[1]):
        #             boxes_to_remove.append(p_annot[2])
        #
        #     points_counts = points_annotations[0]
        #     points_coords = points_annotations[1]
        #     for box_id in boxes_to_remove:
        #         # result1 = np.where(bbox_annot == box_id)
        #         # bbox_annot=np.delete(bbox_annot,result1[0][0],0)
        #         for idx1 in range(len(bbox_annotations)):
        #             if bbox_annotations[idx1][5] == box_id:
        #                 bbox_annotations = np.delete(bbox_annotations, [idx1], 0)
        #                 break
        #
        #         # points_counts = np.delete(points_counts, result2[0][0],0)
        #         for idx2 in range(len(points_counts)):
        #             if points_counts[idx2][2] == box_id:
        #                 points_counts = np.delete(points_counts, [idx2], 0)
        #                 break
        #
        #     points_annotations = (points_counts, points_coords)

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
                        #bbox_id +=1
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
                        if not self.is_roots_2:
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

                        bbox_id += 1


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

            if (config.General.NETWORK_TYPE == config.NetworkType.detection_and_estimation or config.General.filter_empty_bbox):
                result_bbox, result_points = self.points_to_bbox(result_bbox, result_points)

        return result_bbox, result_points

    def points_to_bbox(self, result_bbox, result_points):
        img_to_remove = []

        for img in result_points.keys():
            if len(result_points[img])>0:
                for i in range(len(result_points[img])):
                    x1_p,y1_p = result_points[img][i]['x'], result_points[img][i]['y']

                    for j in range(len(result_bbox[img])):
                        x1,y1,x2,y2 = result_bbox[img][j]['x1'], result_bbox[img][j]['y1'],result_bbox[img][j]['x2'],result_bbox[img][j]['y2']
                        if x1_p >= x1 and x1_p <= x2 and y1_p >= y1 and y1_p <= y2:
                            result_points[img][i]['bbox_id'] = result_bbox[img][j]['bbox_id']
                            result_bbox[img][j]['points'].append(result_points[img][i])
                            break
            elif config.General.filter_empty_bbox:
                # img is an empty image
                img_to_remove.append(img)

        for img in img_to_remove:
            del result_points[img]
            del result_bbox[img]

        for img in result_bbox.keys():
            boxes_to_remove = []
            for j in range(len(result_bbox[img])):
                if len(result_bbox[img][j]['points'])>0:
                    result_bbox[img][j]['points_count'] = len(result_bbox[img][j]['points'])
                    result_bbox[img][j]['points_class'] = result_points[img][0]['points_class']

                elif config.General.filter_empty_bbox:
                    boxes_to_remove.append(j)

            # delete empty bounding boxes
            boxes_to_remove = set(boxes_to_remove)
            result_bbox[img] = [
                box for i, box in enumerate(result_bbox[img])
                if i not in boxes_to_remove
            ]



        return result_bbox, result_points

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


########################################################################################################################


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


def LCC_collater(data):

    imgs = [s['img'] for s in data]

    if 'annot' in data[0].keys():
        have_GT = True
        annots = [s['annot'] for s in data]
    else:
        have_GT = False

    scales = [s['scale'] for s in data if 'scale' in s.keys()]

    batch_size = len(imgs)

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]

    if have_GT:
        for i in range(len(annots)):
            annots[i][0] = np.asarray([annots[i][0][0:4]], dtype=np.float64)


    for i in range(len(imgs)):
        imgs[i] = torch.tensor(imgs[i]) #, dtype=torch.double, device=torch.device('cuda:0'))

        if have_GT:
            for j in range(len(annots[i])):
                annots[i][j] = torch.tensor(annots[i][j]) #, dtype=torch.double, device=torch.device('cuda:0'))


    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    if have_GT:

        if len(annots) > 1:
            max_num_annots = max(len(annot) for annot in annots)

            if max_num_annots > 0:
                map_widths = [int(annot[1].shape[0]) for annot in annots]
                map_heights = [int(annot[1].shape[1]) for annot in annots]
                max_map_width = np.array(map_widths).max()
                max_map_height = np.array(map_heights).max()

                annot_padded = torch.ones((len(annots), max_num_annots, max_map_width, max_map_height)) * -1

            if max_num_annots > 0:
                    for idx, annot in enumerate(annots):
                        if len(annot) > 0:
                            annot[0] = torch.ones((int(annot[1].shape[0]), int(annot[1].shape[1])), dtype=torch.double) * annot[0].data
                            annot_padded[idx, 0, :annot[0].shape[0], :annot[0].shape[1]] = annot[0]
                            for j in range(1, len(annot)):
                                annot_padded[idx, j, :annot[j].shape[0], :annot[j].shape[1]] = annot[j]


            return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
        else:
            new_annots = []

            for i in range(len(annots[0])):
                ann = []
                for j in range(len(annots)):
                    annots_temp = annots[j][i].float()
                    a=torch.unsqueeze(annots_temp, dim=0)
                    ann.append(a)

                new_annots.append(torch.cat(ann, dim=0))

            return {'img': padded_imgs, 'annot': new_annots, 'scale': scales}

    else:
        return {'img': padded_imgs}


def kcsv_collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    # scale_rows = [s['scale_rows'] for s in data]
    # scale_cols = [s['scale_cols'] for s in data]
    scale = [s['scale'] for s in data]

    names = [s['img_name'] for s in data]

    # if config.General.NETWORK_TYPE not in [config.NetworkType.detection, config.NetworkType.detection_and_counting]:
    #     lean_version = data[0]['lean_version']

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    # Turn images and annotations to tensors
    for i in range(len(imgs)):
        imgs[i] = torch.tensor(imgs[i]) #, dtype=torch.double, device=torch.device('cuda:0'))
        # per image annotations:

        if 'points_annot' in annots[i].keys(): # and not config.General.filter_empty_bbox:
            if len(annots[i]['points_annot']) > 0:
                points_annot = annots[i]['points_annot']
            else:
                points_annot = None
        else:
            points_annot = None

        if len(annots[0])>0:
            if len(annots[i]['bbox_annot']) > 0:
                bbox_annot = annots[i]['bbox_annot']
            else:
                bbox_annot = None
        else:
            bbox_annot = None

        if bbox_annot is not None:
            annots[i]['bbox_annot'] = torch.tensor(bbox_annot)

        if points_annot is not None:
            points_annot[0] = torch.tensor(points_annot[0])
            if config.General.NETWORK_TYPE not in (config.NetworkType.detection_and_estimation, config.NetworkType.detection):
                points_annot[1] = torch.tensor(points_annot[1])
                points_annot[2] = torch.tensor(points_annot[2])
                points_annot[3] = torch.tensor(points_annot[3])
                points_annot[4] = torch.tensor(points_annot[4])
                points_annot[5] = torch.tensor(points_annot[5])
            else:
                points_annot[1] = torch.tensor(points_annot[1])


    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    ######################################################################################################
    # bbox annots
    ######################################################################################################

    if bbox_annot is not None:
        bbox_max_num_annots = max(img_annot['bbox_annot'].shape[0] for img_annot in annots)
    else:
        bbox_max_num_annots = 0

    if bbox_max_num_annots > 0:

        bbox_annot_padded = torch.ones((len(annots), bbox_max_num_annots, 6)) * -1

        for idx, annot in enumerate(annots):
            # print(annot.shape)
            if annot['bbox_annot'].shape[0] > 0:
                bbox_annot_padded[idx, :annot['bbox_annot'].shape[0], :] = annot['bbox_annot']
    else:
        bbox_annot_padded = torch.ones((len(annots), 1, 6)) * -1

    if points_annot is None:
        return {'img': padded_imgs, 'bbox_annot': bbox_annot_padded, 'scale': scale, 'img_name':names}

    #######################################################################################################
    # points annotations
    #######################################################################################################


    points_annot = []
    for i in range(len(annots)):
       points_annot.append(annots[i]['points_annot'])

    return {'img': padded_imgs, 'bbox_annot': bbox_annot_padded, 'points_annot': points_annot,
            'scale': scale, 'img_name': names}


def kcsv_collater_2(data):
    # Turn images and annotations to tensors

    imgs = [s['img'] for s in data]

    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[0]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)

    for i in range(batch_size):
        # imgs[i] = torch.tensor(imgs[i]) #, dtype=torch.double, device=torch.device('cuda:0'))
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    # scale = [s['scale'] for s in data]
    # lean_version = data[0]['lean_version']

    if 'annot' in data[0].keys():
        annots = [s['annot'] for s in data]

        # per image annotations:
        new_annots = []
        anns_number = len(annots[0])
        for i in range(anns_number): #range(6):
            ann = []
            for j in range(batch_size):
                if i==0:
                    annots_temp = torch.tensor(annots[j][i]).float()

                else:
                    annots_temp = torch.tensor(annots[j][i]) #annots[j][i]
                a = torch.unsqueeze(annots_temp, dim=0)
                ann.append(a)

            if i==0:
                new_annots.append(torch.cat(ann, dim=0))
            else:
                map_heights = [int(annot.shape[1]) for annot in ann]
                map_widths = [int(annot.shape[2]) for annot in ann]
                max_map_width = np.array(map_widths).max()
                max_map_height = np.array(map_heights).max()
                annot_padded = torch.ones((len(ann), max_map_height, max_map_width)) * -1
                #import matplotlib.pyplot as plt
                #plt.imsave('vis' + '/' + 'ann1' + '_Relu.png', annotation_map_1)

                for idx, annot in enumerate(ann):
                    #temp = torch.ones((int(annot.shape[1]), int(annot.shape[2])), dtype=torch.float).cpu()* annot[0].data
                    #annot_padded[idx, :annot.shape[1], :annot.shape[2]] = temp
                    annot_padded[idx, :annot.shape[1], :annot.shape[2]] = annot[0]

                new_annots.append(annot_padded)

        if "roots_annot" not in data[0].keys():
            return {'img': padded_imgs, 'points_annot': new_annots }

        else: # ToDo - remove roots_annot
            roots_annot = [torch.unsqueeze(s['roots_annot'], dim=0) for s in data]
            new_annots.append(torch.cat(roots_annot, dim=0))

            #roots_annot = [s['roots_annot'] for s in data]
            # gt_box_color = []
            # roots_length = []
            # roots_dia = []
            # for i in range(3):
            #     ann = []
            #     for j in range(batch_size):
            #         annots_temp = torch.tensor(roots_annot[j][i])  # annots[j][i]
            #         a = torch.unsqueeze(annots_temp, dim=0).float()
            #         if i==0:
            #            gt_box_color.append(a)
            #         elif i==1:
            #             roots_length.append(a)
            #         elif i==2:
            #             roots_dia.append(a)
            #
            # new_annots.append(torch.cat(gt_box_color, dim=0))
            # new_annots.append(torch.cat(roots_length, dim=0))
            # new_annots.append(torch.cat(roots_dia, dim=0))

            if 'gt_box_maps' in data[0].keys():
                annots = [s['gt_box_maps'] for s in data]

                # per image annotations:
                box_annots = []

                for i in range(5):
                    ann = []
                    for j in range(batch_size):
                        if i == 0:
                            annots_temp = torch.tensor(annots[j][i]).float()

                        else:
                            annots_temp = torch.tensor(annots[j][i])  # annots[j][i]
                        a = torch.unsqueeze(annots_temp, dim=0)
                        ann.append(a)

                    if i == 0:
                        box_annots.append(torch.cat(ann, dim=0))
                    else:
                        map_heights = [int(annot.shape[1]) for annot in ann]
                        map_widths = [int(annot.shape[2]) for annot in ann]
                        max_map_width = np.array(map_widths).max()
                        max_map_height = np.array(map_heights).max()
                        annot_padded = torch.ones((len(ann), max_map_height, max_map_width)) * -1
                        # import matplotlib.pyplot as plt
                        # plt.imsave('vis' + '/' + 'ann1' + '_Relu.png', annotation_map_1)

                        for idx, annot in enumerate(ann):
                            # temp = torch.ones((int(annot.shape[1]), int(annot.shape[2])), dtype=torch.float).cpu()* annot[0].data
                            # annot_padded[idx, :annot.shape[1], :annot.shape[2]] = temp
                            annot_padded[idx, :annot.shape[1], :annot.shape[2]] = annot[0]

                        box_annots.append(annot_padded)


                return {'img': padded_imgs, 'points_annot': new_annots, 'box_maps_annot': box_annots}

            return {'img': padded_imgs, 'points_annot': new_annots}

    else:
        return {'img': padded_imgs}


########################################################################################################################


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, ann_type=None, min_side=608, max_side=1024):

        self.min_side = min_side
        self.max_side = max_side
        self.ann_type = ann_type


    def __call__(self, sample): #, min_side=608, max_side=1024):

        if len(sample.keys()) > 1:
            image, annots = sample['img'], sample['annot']
        else:
            image = sample['img']

        orig_rows, orig_cols, cns = image.shape

        smallest_side = min(orig_rows, orig_cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(orig_rows, orig_cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(orig_rows*scale)), int(round((orig_cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        if len(sample.keys()) > 1:
            if self.ann_type is not None: # for the separate detection\counting tasks from the 'coco'\'csv_LCC' dataloaders

                if self.ann_type == 'bbox':
                    annots[:, :4] *= scale

                    return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

                elif self.ann_type == 'count':
                    #image_scale_rows = (rows + pad_w) / orig_rows #(rows + pad_w) / rows
                    #image_scale_cols = (cols + pad_h) / orig_cols #(cols + pad_h) / cols

                    if len(annots) == 1 or len(annots[1])==0:
                        annotations_group_num_of_leaves = annots
                        annotations = [annotations_group_num_of_leaves]

                    elif len(annots) == 2 :
                        annotations_group_num_of_leaves, annotations_group_leaves_center = annots

                        annotations_group_leaves_center[:, 0] *= scale #scale #image_scale_rows
                        annotations_group_leaves_center[:, 1] *= scale #scale #image_scale_cols
                        #annotations_group_leaves_center[:,:2] *= scale

                        annotations = [annotations_group_num_of_leaves, annotations_group_leaves_center]

                return {'img': new_image, 'annot': annotations}

            else:

                # image_scale_rows = (rows + pad_w) / orig_rows #(rows + pad_w) / rows
                # image_scale_cols = (cols + pad_h) / orig_cols #(cols + pad_h) / cols

                if 'bbox_annot' in annots.keys():
                    if len(annots['bbox_annot']) > 0:
                        annots['bbox_annot'][:, 0] *= scale #image_scale_rows
                        annots['bbox_annot'][:, 1] *= scale #image_scale_cols
                        annots['bbox_annot'][:, 2] *= scale #image_scale_rows
                        annots['bbox_annot'][:, 3] *= scale #image_scale_cols

                    #annots['bbox_annot'][:, :4] *= scale

                if 'points_annot' in annots.keys():
                    if len(annots['points_annot']) > 0:
                        annotations_num_of_points, annotations_points_center = annots['points_annot']
                        annotations_points_center[:, 0] *= scale #image_scale_rows
                        annotations_points_center[:, 1] *= scale #image_scale_cols

                        annots['points_annot'] = [annotations_num_of_points, annotations_points_center]

                return {'img': new_image, 'annot': annots, 'scale': scale}

        else:
            return {'img': new_image}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, pre_process=None):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

        self.pre_process = pre_process


    def __call__(self, sample):

        if len(sample.keys())>1:
            image, annots = sample['img'], sample['annot']
        else:
            image = sample['img']

        # The regular 'coco' version
        if self.pre_process is None:
            if len(sample.keys()) > 1:
                return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
            else:
                return {'img': ((image.astype(np.float32) - self.mean) / self.std)}


        # for counting or per-object estimation
        if self.pre_process == "torch_like":
            image = image.astype(np.float32) / 255.0
            if len(sample.keys()) > 1:
                return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
            else:
                return {'img': (image.astype(np.float32) - self.mean) / self.std}



        elif self.pre_process == "keras_like":
            image = image.astype(float)
            image[..., 0] -= 103.939
            image[..., 1] -= 116.779
            image[..., 2] -= 123.68

            if len(sample.keys()) > 1:
                return {'img':image, 'annot': annots}
            else:
                return {'img': image}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

#Todo - UnNormalizer for 'keras_like' version

class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last, do_shuffle = True):
        self.do_shuffle = do_shuffle
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        if self.do_shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


class Resizer_2(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, ann_type=None, min_side=608, max_side=1024):

        self.min_side = min_side
        self.max_side = max_side
        self.ann_type = ann_type


    def __call__(self, sample): #, min_side=608, max_side=1024):

        image, annots = sample['img'], sample['annot']

        orig_rows, orig_cols, cns = image.shape

        smallest_side = min(orig_rows, orig_cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(orig_rows, orig_cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image.numpy(), (int(round(orig_rows*scale)), int(round((orig_cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # image_scale_rows = (rows + pad_w) / orig_rows #(rows + pad_w) / rows
        # image_scale_cols = (cols + pad_h) / orig_cols #(cols + pad_h) / cols

        if len(annots)>0:

            annotations_points_center = annots
            annotations_num_of_points = len(annots)

            for i in range(len(annots)):
                annotations_points_center[i]['x'] *= scale
                annotations_points_center[i]['y'] *= scale


            annots = [annotations_num_of_points, annotations_points_center]

        return {'img': new_image, 'annot': annots, 'scale': scale}


class Normalizer_2(object):

    def __init__(self, pre_process=None):
        self.mean = torch.tensor(np.array([[[0.485, 0.456, 0.406]]])).float()
        self.std = torch.tensor(np.array([[[0.229, 0.224, 0.225]]])).float()

        self.pre_process = pre_process


    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        # The regular 'coco' version
        if self.pre_process is None:
            return {'img': ((image - self.mean) / self.std), 'annot': annots}

        # for counting or per-object estimation
        if self.pre_process == "torch_like":

            image = image / 255.0
            return {'img': ((image - self.mean) / self.std), 'annot': annots}

        elif self.pre_process == "keras_like":
            image = image.astype(float)
            image[..., 0] -= 103.939
            image[..., 1] -= 116.779
            image[..., 2] -= 123.68

            return {'img':image, 'annot': annots}
