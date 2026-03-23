import os
import shutil
import numpy as np
import csv
import cv2
import random

from os.path import exists
########################################################################################################################
current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

random.seed(0)
np.random.seed(0)

########################################################################################################################

# have the images, need to get corrected files


# define paths

dataset_name = "Tomato 2019" #"Melon 2019" #"Tomato 2019" #"Pepper 2021" #"melon 2019" #"Corn 2020" #"Tomato 2020"
data_path = os.path.join("D:\\Faina\\roots_project","manual_camera", "13_5_22\\Correction after visual inspection_20220513", dataset_name) #"manual_camera\\5_5_22\\Corrected annotation"

# "tomato 2020- splitted_data3" , "corn 2020- splitted data" , "splitted_data3-melon 2019" , "pepperm2021-splitted_data"
noRoot_img_path = os.path.join(data_path, "pepperm2021-splitted_data" , "processed_noRoot")
withRoot_img_path = os.path.join(data_path, "pepperm2021-splitted_data", "processed_withRoot")

noRoot_file = os.path.join(noRoot_img_path, "processed_noRoot.csv")
withRoot_file = os.path.join(withRoot_img_path, "pepper_corrected_processed_withRoot.csv")

########################################################################################################################
# if true, saved the cropped version of the images

save_cropped = True
########################################################################################################################

Train_dir = os.path.join(data_path, "sub_Train")
Val_dir = os.path.join(data_path, "sub_Val")
Test_dir = os.path.join(data_path, "sub_Test")

# output splitted files
Train_csv_file = os.path.join(Train_dir, "Train.csv")
Val_csv_file = os.path.join(Val_dir, "Val.csv")
Test_csv_file = os.path.join(Test_dir, "Test.csv")

Train_rows = []
Val_rows = []
Test_rows = []

for file in [noRoot_file, withRoot_file]:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if "noRoot" in file:
            current_dir = noRoot_img_path
        else:
            current_dir = withRoot_img_path

        for row in csv_reader:
            img_name = row[0]
            if exists(os.path.join(Train_dir, img_name)):
                Train_rows.append(row)
            elif exists(os.path.join(Val_dir, img_name)):
                Val_rows.append(row)
            elif exists(os.path.join(Test_dir, img_name)):
                Test_rows.append(row)



# print files

f_Train = open(Train_csv_file, 'w', newline='')
with f_Train:
    writer = csv.writer(f_Train)
    for row in Train_rows:
        writer.writerow(row)

f_Val = open(Val_csv_file, 'w', newline='')
with f_Val:
    writer = csv.writer(f_Val)
    for row in Val_rows:
        writer.writerow(row)


f_Test = open(Test_csv_file, 'w', newline='')
with f_Test:
    writer = csv.writer(f_Test)
    for row in Test_rows:
        writer.writerow(row)




