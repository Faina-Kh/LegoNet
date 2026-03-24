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

# define paths

dataset_name = "Tomato 2019" #"Melon 2018" #"Tomato 2019" #"Pepper 2021" #"melon 2019" #"Corn 2020" #"Tomato 2020"
data_path = os.path.join("D:\Faina\\roots_project", "manual_camera", dataset_name)
                         # "Autocam image for CNN model", "Images for training\\Training dataset_automated camera", "Training dataset_Rootfly")
#os.path.join("D:\\Faina\\roots_project", "manual_camera", "13_5_22\\Correction after visual inspection_20220513", dataset_name)

# "tomato 2020- splitted_data3" , "corn 2020- splitted data" , "splitted_data3-melon 2019" , "pepperm2021-splitted_data"
# noRoot_img_path = os.path.join(data_path, "pepperm2021-splitted_data" , "processed_noRoot")
# withRoot_img_path = os.path.join(data_path, "pepperm2021-splitted_data", "processed_withRoot")
#
# noRoot_file = os.path.join(noRoot_img_path, "processed_noRoot.csv")
# withRoot_file = os.path.join(withRoot_img_path, "pepper_corrected_processed_withRoot.csv")


img_path = data_path  #os.path.join("D:\\Faina\\roots_project", "manual_camera", "13_5_22\\Correction after visual inspection_20220513", "images", dataset_name)

file_path = os.path.join(data_path, "TRL.csv" ) #"new_auto_cam_TRL.csv")  #dataset_name+"_corrected.csv")



########################################################################################################################
# if true, saved the cropped version of the images

save_cropped = False

########################################################################################################################

Train_dir = os.path.join(data_path, "sub_Train")
Val_dir = os.path.join(data_path, "sub_Val")
Test_dir = os.path.join(data_path, "sub_Test")

# create folders if don't exist
os.makedirs(Train_dir, exist_ok=True)
os.makedirs(Val_dir, exist_ok=True)
os.makedirs(Test_dir, exist_ok=True)

# splitted files
Train_csv_file = os.path.join(Train_dir, "Train.csv")
Val_csv_file = os.path.join(Val_dir, "Val.csv")
Test_csv_file = os.path.join(Test_dir, "Test.csv")

Train_rows = []
Val_rows = []
Test_rows = []

for file in [file_path]: #[noRoot_file, withRoot_file]:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # if "noRoot" in file:
        #     current_dir = noRoot_img_path
        # else:
        #     current_dir = withRoot_img_path

        current_dir = img_path

        for row in csv_reader:
            img_name = row[0]
            img_path = os.path.join(current_dir, img_name)

            if exists(img_path):

                if save_cropped:
                    image = cv2.imread(img_path)
                    image = image[21:471, 16:, :]

                # choose randomly were to assign the image
                rnd = np.random.random()
                # copy images
                if rnd <= 0.7:
                    Train_rows.append(row)
                    dst_img_file = os.path.join(Train_dir, img_name)
                    if save_cropped:
                        cv2.imwrite(dst_img_file, image)
                    else:
                        shutil.copyfile(img_path, dst_img_file)

                elif 0.7< rnd <= 0.8:
                    Val_rows.append(row)
                    dst_img_file = os.path.join(Val_dir, img_name)
                    if save_cropped:
                        cv2.imwrite(dst_img_file, image)
                    else:
                        shutil.copyfile(img_path, dst_img_file)

                else:
                    Test_rows.append(row)
                    dst_img_file = os.path.join(Test_dir, img_name)
                    if save_cropped:
                        cv2.imwrite(dst_img_file, image)
                    else:
                        shutil.copyfile(img_path, dst_img_file)


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




