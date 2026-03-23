import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

current_gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))


# storagePath
import paths
myStoragePath = paths.STORAGE_PATH
myDatasetsPath = paths.DATASETS_PATH
myExpResultsPath = paths.EXP_RESULTS_PATH
myModelsPath = paths.MODELS_PATH

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_conf=  ["0.05","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
detect_file= "detections_data_any_crop.csv"
not_found_file="not_found_gt_count.csv"

dir_num= "40" #"50"
current_dir= "Splits\\Plot 128 splits\\plot 128_split " + dir_num + " perc"

read_val_names = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", current_dir, "plot 128_val_names_" + dir_num + ".csv")
read_test_names = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", current_dir, "plot 128_test_names_" + dir_num + ".csv")

for conf in all_conf:

    read_detect_file = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", "plot 128", 'conf ' + conf, detect_file)
    read_unfound_file = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", "plot 128", 'conf ' + conf, not_found_file)

    val_path = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", current_dir, "again_val", 'conf ' + conf)
    test_path = os.path.join("C:\\Users\\Aragorn\\Desktop", "Robust new", "3.try again", current_dir, "again_test", 'conf ' + conf)

    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    write_val_detect_file = os.path.join(val_path,  detect_file)
    write_val_unfound_file = os.path.join(val_path, not_found_file)

    write_test_detect_file = os.path.join(test_path, detect_file)
    write_test_unfound_file = os.path.join(test_path, not_found_file)

    all_detections = pd.read_csv(read_detect_file).to_numpy()
    unfound = pd.read_csv(read_unfound_file).to_numpy()

    val_names = pd.read_csv(read_val_names,  header=None).to_numpy()
    test_names = pd.read_csv(read_test_names,  header=None).to_numpy()

    csv_columns = ["img","pred","gt_count",	"label","score"]

    # write val files
    f1 = open(write_val_detect_file, 'a', newline='')
    with f1:
        writer1 = csv.writer(f1)
        writer1.writerow(csv_columns)
        for i in range(len(all_detections)):
            current_img = all_detections[i][0]
            myrow = all_detections[i]
            if current_img in val_names:
                writer1.writerow(myrow)

    f2 = open(write_val_unfound_file, 'a', newline='')
    with f2:
        writer2 = csv.writer(f2)
        writer2.writerow(csv_columns)
        for i in range(len(unfound)):
            current_img = unfound[i][0]
            myrow = unfound[i]
            if current_img in val_names:
                writer2.writerow(myrow)

    # write test files
    f3 = open(write_test_detect_file, 'a', newline='')
    with f3:
        writer3 = csv.writer(f3)
        writer3.writerow(csv_columns)
        for i in range(len(all_detections)):
            current_img = all_detections[i][0]
            myrow = all_detections[i]
            if current_img in test_names:
                writer3.writerow(myrow)

    f4 = open(write_test_unfound_file, 'a', newline='')
    with f4:
        writer4 = csv.writer(f4)
        writer4.writerow(csv_columns)
        for i in range(len(unfound)):
            current_img = unfound[i][0]
            myrow = unfound[i]
            if current_img in test_names:
                writer4.writerow(myrow)
