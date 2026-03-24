import json
import csv
import pandas as pd
import numpy as np

def load_via_annotations_csv(csv_path):

    via_df = pd.read_csv(csv_path)
    images_names_complete_df = pd.DataFrame(via_df.iloc[:, 0])
    images_names = pd.unique(via_df.iloc[:, 0])
    annotations = pd.DataFrame([via_df.iloc[:, 1], via_df.iloc[:, 2]])

    for image_info in range(len(images_names)):
        current_image_name = images_names[image_info]
        number_of_centers = 0
        annotations_indices = np.where(images_names_complete_df == current_image_name)
        annotations_for_image = annotations.iloc[:, annotations_indices[0]]
        for anno in range(annotations_for_image.shape[1]):
            x = annotations_for_image.iloc[0, anno]
            y = annotations_for_image.iloc[1, anno]

            number_of_centers += 1

    x = 0

def getCentroid(x, y):

    A = 0
    for i in range(0,len(x)-1):
        A += (x[i]*y[i+1] - x[i+1]*y[i])

    A = 0.5*A

    Cx = 0
    Cy = 0

    for i in range(0, len(x) - 1):
        Cx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        Cy += (y[i] + y[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i])

    Cx = Cx / (6*A)
    Cy = Cy / (6*A)

    return [Cx, Cy]

def readVia(path):

    with open(path) as json_file:
        data = json.load(json_file)

        for key in data.keys():
            items = data[key]
            shape = items["regions"]
            for attr in shape:

                if attr["shape_attributes"]["name"] == "polygon":
                    point_dict = {}

                    x = attr["shape_attributes"]["all_points_x"]
                    y = attr["shape_attributes"]["all_points_y"]

                    C = getCentroid(x,y)
                    point_dict["name"] = "point"
                    point_dict["cx"] = int(C[0])
                    point_dict["cy"] = int(C[1])

                    temp_shape = {}
                    temp_shape["shape_attributes"] = point_dict
                    temp_shape["region_attributes"] = attr["region_attributes"]

                    shape.append(temp_shape)

    with open('data_new.json', 'w') as outfile:
        json.dump(data, outfile)

def main():

    # path = "C:\\Users\\stas\\Desktop\\bunch_Seg-20200218T091708Z-001\\bunch_Seg\\via_region_data.json"
    # readVia(path)

    path = "annotations_119.csv"
    load_via_annotations_csv(path)

if __name__ == '__main__':
    main()

def getCentroid(x, y):

    A = 0
    for i in range(0,len(x)-1):
        A += (x[i]*y[i+1] - x[i+1]*y[i])

    A = 0.5*A

    Cx = 0
    Cy = 0

    for i in range(0, len(x) - 1):
        Cx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        Cy += (y[i] + y[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i])

    Cx = Cx / (6*A)
    Cy = Cy / (6*A)

    return [Cx, Cy]


import pandas as pd
import os
import json
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import re
import csv
from os import listdir
from os.path import isfile, join
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)
import paths
from phenomics import *


def read_json(jfile):
    with open(jfile, 'r') as f:
        return json.load(f)

def is_keypoint_in_polygon(keypoint,polygon_points):
    point = Point(keypoint[0], keypoint[1])
    polygon = Polygon([[p.x, p.y] for p in polygon_points])
    return polygon.contains(point)

def is_keypoint_in_polygon(keypoint,polygon_points_x, polygon_points_y):
    point = Point(keypoint[0], keypoint[1])
    polygon_points = []
    for i in range(len(polygon_points_x)):
        polygon_points.append([polygon_points_x[i], polygon_points_y[i]])
    polygon = Polygon(polygon_points)
    return polygon.contains(point)

def show_bboxes(json_anno_path, images_dir, dataset_images_dir_path):
    '''

    :param json_anno_path:
    :param images_dir:
    :param dataset_images_dir_path:
    :return:
    '''
    j = read_json(json_anno_path)
    images = j['images']
    annotations = pd.DataFrame(j['annotations'])

    for image_info in images:
        number_of_bbs = 0
        img = cv2.imread(os.path.join(images_dir, image_info['file_name']))
        annotations_for_image = annotations.loc[annotations['image_id'] == image_info['id']]
        for anno in annotations_for_image.iterrows():
            x1, y1, w, h = anno[1]['bbox']
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),3)
            number_of_bbs+=1
        cv2.imshow('', img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(dataset_images_dir_path, 'annotations_drawing_detection', image_info['file_name']), img)
        print(image_info['file_name'] + ' has ' + str(number_of_bbs) + ' bboxes')

def show_centers(csv_anno_path, images_dir, dataset_images_dir_path):
    '''

    :param csv_anno_path: path to annotations file
    :param images_dir: path to images directory
    :param dataset_images_dir_path: where to save the drawings
    :return:
    '''
    c = pd.read_csv(csv_anno_path)
    images_names_complete_df = pd.DataFrame(c.iloc[:,0])
    images_names = np.unique(c.iloc[:,0])
    annotations = pd.DataFrame([c.iloc[:,1], c.iloc[:,2]])

    for image_info in range(len(images_names)):
        current_image_name = images_names[image_info]
        number_of_centers = 0
        img = cv2.imread(os.path.join(images_dir, current_image_name))
        annotations_indices = np.where(images_names_complete_df == current_image_name)
        annotations_for_image = annotations.iloc[:, annotations_indices[0]]
        for anno in range(annotations_for_image.shape[1]):
            x = annotations_for_image.iloc[0,anno]
            y = annotations_for_image.iloc[1,anno]
            cv2.circle(img, (x,y), radius = 2,color = (0,255,0),thickness = 3)
            number_of_centers+=1
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(dataset_images_dir_path, 'annotations_drawing_counting', current_image_name), img)
        print(current_image_name + ' has ' + str(number_of_centers) + ' centers')


def create_detection_dataset(annotations_by_image, image_ids, ids_and_names, dataset_images_dir_path, output_file_path, vis_anno = 0):
    splitted_images_dir_path = os.path.join(dataset_images_dir_path[:-6], 'splitted_images_detection')
    patch_id = 0
    annotation_id = 0
    images = []
    annotations = []
    for i in range(len(annotations_by_image)):
        patchs_and_coor = {}
        image_annotations = annotations_by_image[i]
        original_image = cv2.imread(os.path.join(dataset_images_dir_path, ids_and_names[image_ids[i]]))
        for index, anno in image_annotations.iterrows():
            if anno['annotation_type'] == 'rectangle':
                if patch_id < 10:
                    current_patch_name = 'IMG_000' + str(patch_id) + '.jpg'
                elif patch_id < 100:
                    current_patch_name = 'IMG_00' + str(patch_id) + '.jpg'
                elif patch_id < 1000:
                    current_patch_name = 'IMG_0' + str(patch_id) + '.jpg'
                else:
                    current_patch_name = 'IMG_' + str(patch_id) + '.jpg'
                current_patch_path = os.path.join(splitted_images_dir_path, current_patch_name)
                coordinates = re.sub(r'[^a-zA-Z0-9\,._-]', '', anno.get('annotations'))
                coordinates = [x for x in coordinates.split(',')]
                x = int(float(coordinates[0][1:]))-10
                y = int(float(coordinates[1][1:]))-10
                height = int(float(coordinates[2][6:]))+20
                width = int(float(coordinates[3][5:]))+20
                image_patch = original_image[y:y+height, x:x+width, :]
                cv2.imwrite(current_patch_path, image_patch)
                patchs_and_coor[patch_id] = [x, y, width, height]
                images.append({'file_name': current_patch_name, "width": width, "height": height, "id": patch_id})
                patch_id += 1
            elif anno['annotation_type'] == 'polygon':
                coordinates = re.sub(r'[^a-zA-Z0-9\,._-]', '', anno.get('annotations'))
                number_of_points = coordinates.count('x')
                all_points_x = []
                all_points_y = []
                for j in range(0, number_of_points*2 , 2):
                    all_points_x.append(float(coordinates.split(',')[j][1:]))
                    all_points_y.append(float(coordinates.split(',')[j + 1][1:]))
                all_points = np.reshape([[all_points_x, all_points_y]], 2 * number_of_points, order='F').tolist()

                min_x = np.min(all_points_x)
                min_y = np.min(all_points_y)
                max_x = np.max(all_points_x)
                max_y = np.max(all_points_y)

                for i in patchs_and_coor.keys():
                    patch = patchs_and_coor.get(i)
                    if (min_x > patch[0]) and (min_y > patch[1]) and (max_x < patch[0]+patch[2]) and (max_y < patch[1]+patch[3]):
                        area =  0.5 * np.abs(np.dot(all_points_x, np.roll(all_points_y, 1)) - np.dot(all_points_y, np.roll(all_points_x, 1)))
                        bbox = [min_x - patch[0], min_y - patch[1], max_x-min_x, max_y-min_y]
                        break
                annotations.append({'id': annotation_id, 'image_id': i, 'iscrowd': 0, 'category_id': 1, 'segmentation': all_points, 'area': area, 'bbox': bbox})
                annotation_id +=1
    categories = [{'supercategory': '', 'name': '', 'id': 1}]
    data = {'categories': categories, 'images': images, 'annotations': annotations}
    with open(output_file_path, 'w') as fp:
        json.dump({"categories": data["categories"], "images": data["images"], "annotations": data['annotations']}, fp)

    if(vis_anno):
        show_bboxes(output_file_path, splitted_images_dir_path, dataset_images_dir_path[:-6])

def create_counting_dataset(annotations_by_image, image_ids, ids_and_names, dataset_images_dir_path, output_file_path_centers, output_file_path_count, vis_anno = 0):
    splitted_images_dir_path = os.path.join(dataset_images_dir_path[:-6], 'splitted_images_counting')

    if not os.path.exists(output_file_path_centers):

        patch_id = 0
        image_patchs = {}
        centers_coors_df_list = []
        centers_count_df_list = []

        for i in range(len(annotations_by_image)):
            patchs_and_coor = {}
            patchs_and_names = {}
            images_patchs_ids = []
            patchs_as_bbox = {}
            image_annotations = annotations_by_image[i]
            original_image = cv2.imread(os.path.join(dataset_images_dir_path, ids_and_names[image_ids[i]]))

            if original_image is None:
                continue

            for index, anno in image_annotations.iterrows():
                if anno['annotation_type'] == 'polygon':
                    if patch_id < 10:
                        current_patch_name = 'IMG_0000' + str(patch_id) + '.jpg'
                    elif patch_id < 100:
                        current_patch_name = 'IMG_000' + str(patch_id) + '.jpg'
                    elif patch_id < 1000:
                        current_patch_name = 'IMG_00' + str(patch_id) + '.jpg'
                    elif patch_id < 10000:
                        current_patch_name = 'IMG_0' + str(patch_id) + '.jpg'
                    else:
                        current_patch_name = 'IMG_' + str(patch_id) + '.jpg'
                    current_patch_path = os.path.join(dataset_images_dir_path[:-6], 'splitted_images_counting',
                                                      current_patch_name)
                    coordinates = re.sub(r'[^a-zA-Z0-9\,._-]', '', anno.get('annotations'))
                    number_of_points = coordinates.count('x')
                    all_points_x = []
                    all_points_y = []
                    for j in range(0, number_of_points * 2, 2):
                        all_points_x.append(float(coordinates.split(',')[j][1:]))
                        all_points_y.append(float(coordinates.split(',')[j + 1][1:]))
                    all_points = np.reshape([[all_points_x, all_points_y]], 2 * number_of_points, order='F').tolist()

                    min_x = int(np.min(all_points_x)) - 2
                    min_y = int(np.min(all_points_y)) - 2
                    max_x = int(np.max(all_points_x)) + 2
                    max_y = int(np.max(all_points_y)) + 2

                    image_patch = original_image[min_y:max_y, min_x:max_x, :]
                    cv2.imwrite(current_patch_path, image_patch)
                    patchs_and_coor[patch_id] = {'x' :all_points_x, 'y': all_points_y}
                    patchs_and_names[patch_id] = current_patch_name
                    patchs_as_bbox[patch_id] = [min_x, min_y, max_x, max_y]
                    images_patchs_ids.append(patch_id)
                    patch_id += 1

            image_patchs[i] = images_patchs_ids

            for j in image_patchs[i]:
                centers_count = 0
                image_name = patchs_and_names[j]
                poly_x = patchs_and_coor[j]['x']
                poly_y = patchs_and_coor[j]['y']

                for index, anno in image_annotations.iterrows():
                    if anno['annotation_type'] == 'point':
                        coordinates = re.sub(r'[^a-zA-Z0-9\,._-]', '', anno.get('annotations'))
                        x = int(float(coordinates.split(',')[0][1:]))
                        y = int(float(coordinates.split(',')[1][1:]))

                        if is_keypoint_in_polygon([x,y], poly_x, poly_y):
                            bbox = patchs_as_bbox[j]
                            centers_coors_df_list.append([image_name, x - bbox[0],y - bbox[1]])
                            centers_count += 1
                centers_count_df_list.append([image_name, centers_count])

        centers_coors_df = pd.DataFrame(centers_coors_df_list, columns=['image_name', 'x', 'y'])
        centers_count_df = pd.DataFrame(centers_count_df_list, columns=['image_name', 'count'])

        centers_coors_df.to_csv(output_file_path_centers, index=False)
        centers_count_df.to_csv(output_file_path_count, index=False)
    if (vis_anno):
        show_centers(output_file_path_centers, splitted_images_dir_path, dataset_images_dir_path[:-6])

def convert_annotations_to_coco(df_annotations,
                                ids_and_names,
                                dataset_images_dir_path,
                                output_file_path_detection_task,
                                output_file_path_counting_task_centers,
                                output_file_path_counting_task_count,
                                vis_anno = 0
                                ):

    image_ids = list(ids_and_names.keys())
    annotations_by_image = []

    for i in range(len(ids_and_names)):
        mask = df_annotations['image_id'] == image_ids[i]
        annotations_by_image.append(df_annotations[mask])

    # create_detection_dataset(annotations_by_image, image_ids, ids_and_names, dataset_images_dir_path, output_file_path_detection_task, vis_anno)

    create_counting_dataset(annotations_by_image, image_ids, ids_and_names, dataset_images_dir_path, output_file_path_counting_task_centers, output_file_path_counting_task_count, vis_anno)

def split_train_validation_test():
    pass

def map_image_id_to_frame(frame_ids):
    p = Phenomics()
    p.login('guyf', 'amtgO1923476')
    results = {}
    for frame in frame_ids:
        res = p.get_image_by_frame(str(frame))
        if len(res) == 0:
            print('There are no images for this task!')
        else:
            results[res[0]['image_id']] = res[0]['image_uri'][-12:]
    return results

if __name__ == '__main__':
    storage_path = os.path.join(paths.DATASETS_PATH, 'Phenomics data')

    annotations_task_id = 119
    experiments_id = 43
    mode = 'images'
    dataset_name = 'wheat_plants'
    vis_anno = 1

    DATASET_DIR = os.path.join(storage_path, dataset_name, 'exp_{}'.format(str(experiments_id)))
    dataset_images_dir_path = join(DATASET_DIR,'images')
    ANNOTATION_PATH = join(DATASET_DIR,'annotations_{}.csv'.format(str(annotations_task_id)))


    df_annotations = pd.read_csv(ANNOTATION_PATH,encoding = 'utf8',index_col=0)

    frame_ids = df_annotations['frame_id'].unique()
    ids_and_names = map_image_id_to_frame(frame_ids)

    convert_to_file_name = "instances_{}_{}.json".format(str(annotations_task_id),mode)
    if not os.path.exists(os.path.join(DATASET_DIR, "annotations_detection")):
        os.makedirs(os.path.join(DATASET_DIR, "annotations_detection"))
    output_file_path_detection_task = os.path.join(DATASET_DIR, "annotations_detection", convert_to_file_name)

    if not os.path.exists(os.path.join(DATASET_DIR, "annotations_counting")):
        os.makedirs(os.path.join(DATASET_DIR, "annotations_counting"))

    counting_centers_file_name = 'WP.csv' #stands for wheat plants
    counting_count_file_name = 'WP_wheat_count.csv'
    output_file_path_counting_task_centers = os.path.join(DATASET_DIR, "annotations_counting", counting_centers_file_name)
    output_file_path_counting_task_count = os.path.join(DATASET_DIR, "annotations_counting", counting_count_file_name)
    convert_annotations_to_coco(df_annotations,
                                ids_and_names,
                                dataset_images_dir_path,
                                output_file_path_detection_task,
                                output_file_path_counting_task_centers,
                                output_file_path_counting_task_count,
                                vis_anno
                                )

