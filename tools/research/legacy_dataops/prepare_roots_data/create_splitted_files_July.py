import os
import shutil
import numpy as np
import csv
import cv2
import random
from os.path import exists
import shutil
import json

########################################################################################################################
current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

random.seed(0)
np.random.seed(0)

##############################################################################################################################################

# have the images, need to get corrected files

# define paths

dataset_name = "" #"Tomato 2020"  #"Corn 2020" #"Melon 2019" #"Tomato 2020" #"Tomato 2019" #"Pepper 2021"  # #"Tomato 2020" # "Melon 2018"
data_path = os.path.join("D:/Faina/roots_project", "Dataset for root color model\\Selected images from Brian_Grapevine",
                         "Rootfly_subfolders_Grapevine roots") #os.path.join("D:\\Faina\\roots_project", "manual_camera", "July_22", dataset_name)

prev_path = os.path.join("D:\\Faina\\roots_project", "manual_camera", "13_5_22\\Correction after visual inspection_20220513", dataset_name)

# "tomato 2020- splitted_data3" , "corn 2020- splitted data" , "splitted_data3-melon 2019" , "pepperm2021-splitted_data"
input_TRL = os.path.join(data_path, "TRL.csv") # dataset_name+"_TRL.csv"
input_points = os.path.join(data_path, "pointsOutput.csv") #dataset_name+"_pointsOutput.csv"

txt_path = os.path.join(data_path, "all_data_Diameter_Length_Color.txt")

Train_dir = os.path.join(data_path, "sub_Train")
Val_dir = os.path.join(data_path, "sub_Val")
Test_dir = os.path.join(data_path, "sub_Test")

os.makedirs(Train_dir, exist_ok=True)
os.makedirs(Val_dir, exist_ok=True)
os.makedirs(Test_dir, exist_ok=True)

prev_Train_dir = os.path.join(prev_path, "sub_Train")
prev_Val_dir = os.path.join(prev_path, "sub_Val")
prev_Test_dir = os.path.join(prev_path, "sub_Test")

with_time_format = True
save_cropped = False

############################################################################################################################################

# create splitted files
Train_csv_file = os.path.join(Train_dir, "Train.csv")
Val_csv_file = os.path.join(Val_dir, "Val.csv")
Test_csv_file = os.path.join(Test_dir, "Test.csv")

# splitted output points files
Train_pointsOutput_file = os.path.join(Train_dir, "Train_pointsOutput.csv")
Val_pointsOutput_file = os.path.join(Val_dir, "Val_pointsOutput.csv")
Test_pointsOutput_file = os.path.join(Test_dir, "Test_pointsOutput.csv")

Train_txt = os.path.join(Train_dir, "Train_Dia_Length_Color.txt")
Val_txt = os.path.join(Val_dir, "Val_Dia_Length_Color.txt")
Test_txt = os.path.join(Test_dir, "Test_Dia_Length_Color.txt")

Train_rows = []
Val_rows = []
Test_rows = []

Train_P_rows = []
Val_P_rows = []
Test_P_rows = []

Train_names=[]
Val_names=[]
Test_names=[]

train_roots_dict={}
val_roots_dict={}
test_roots_dict={}
###########################################################################################################################

with open(input_TRL) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = row[0]

        if save_cropped:
            image = cv2.imread(os.path.join(data_path, img_name))
            image = image[21:471, 16:, :]

        if dataset_name == "Corn 2020" and img_name=="CORNGROUP1_T001_L027_2020.05.25_171355_002_ZKN.jpg":
            img_name = "CORNGROUP1_T001_L027_2020.05.25_171505_002_ZKN.jpg"

        if dataset_name == "Melon 2019" and img_name=="MELONJAN_T027_L012_2019.02.17_132727_002_JEE.jpg":
            img_name = "MELONJAN_T027_L012_2019.02.17_132736_002_JEE.jpg"

        if dataset_name == "Pepper 2021":
            if img_name == "PEPPERARAVA_T003_L022_2021.09.27_085412_003_JEE.jpg":
                img_name = "PEPPERARAVA_T003_L022_2021.09.27_085920_003_JEE.jpg"

            if img_name == "PEPPERARAVA_T021_L045_2021.10.24_101208_005_JEE.jpg" :
                img_name = "PEPPERARAVA_T021_L045_2021.10.24_102527_005_JEE.jpg"

        if dataset_name == "Tomato 2020" and img_name == "TOMATO2020_T015_L017_2020.11.08_092442_003_JEE.jpg":
            img_name = "TOMATO2020_T015_L017_2020.11.08_092614_003_JEE.jpg"

        img_path = os.path.join(data_path, row[0])
        if exists(os.path.join(prev_Train_dir, img_name)):
            Train_rows.append(row)
            dst_img_file = os.path.join(Train_dir, row[0])
            if save_cropped:
                cv2.imwrite(dst_img_file, image)
            else:
                shutil.copyfile(img_path, dst_img_file)

            Train_names.append(row[0])

        elif exists(os.path.join(prev_Val_dir, img_name)):
            Val_rows.append(row)
            dst_img_file = os.path.join(Val_dir, row[0])
            if save_cropped:
                cv2.imwrite(dst_img_file, image)
            else:
                shutil.copyfile(img_path, dst_img_file)
            Val_names.append(row[0])

        elif exists(os.path.join(prev_Test_dir, img_name)):
            Test_rows.append(row)
            dst_img_file = os.path.join(Test_dir, row[0])
            if save_cropped:
                cv2.imwrite(dst_img_file, image)
            else:
                shutil.copyfile(img_path, dst_img_file)
            Test_names.append(row[0])

        else: # new image
            rnd = np.random.random()
            # copy images
            if rnd <= 0.7:
                Train_rows.append(row)
                dst_img_file = os.path.join(Train_dir, row[0])
                if save_cropped:
                    cv2.imwrite(dst_img_file, image)
                else:
                    shutil.copyfile(img_path, dst_img_file)

            elif 0.7 < rnd <= 0.8:
                Val_rows.append(row)
                dst_img_file = os.path.join(Val_dir, row[0])
                if save_cropped:
                    cv2.imwrite(dst_img_file, image)
                else:
                    shutil.copyfile(img_path, dst_img_file)
                Val_names.append(row[0])

            else:
                Test_rows.append(row)
                dst_img_file = os.path.join(Test_dir, row[0])
                if save_cropped:
                    cv2.imwrite(dst_img_file, image)
                else:
                    shutil.copyfile(img_path, dst_img_file)
                Test_names.append(row[0])



with open(input_points) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = row[0]
        if img_name in Train_names:
            Train_P_rows.append(row)
        elif img_name in Val_names:
            Val_P_rows.append(row)
        elif img_name in Test_names:
            Test_P_rows.append(row)


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

####################################################################################################################


f_P_Train = open(Train_pointsOutput_file, 'w', newline='')
with f_P_Train:
    writer = csv.writer(f_P_Train)
    for row in Train_P_rows:
        writer.writerow(row)

f_P_Val = open(Val_pointsOutput_file, 'w', newline='')
with f_P_Val:
    writer = csv.writer(f_P_Val)
    for row in Val_P_rows:
        writer.writerow(row)


f_P_Test = open(Test_pointsOutput_file, 'w', newline='')
with f_P_Test:
    writer = csv.writer(f_P_Test)
    for row in Test_P_rows:
        writer.writerow(row)


# reading the data from the file
with open(txt_path) as f:
    data = f.read()

roots_dict = json.loads(data)

for key in roots_dict.keys():
    img_name = roots_dict[key]['processed_name']
    if img_name in Train_names:
        train_roots_dict[key]=roots_dict[key]
    elif img_name in Val_names:
        val_roots_dict[key]=roots_dict[key]
    elif img_name in Test_names:
        test_roots_dict[key]=roots_dict[key]

with open(Train_txt, "w") as outfile:
        json.dump(train_roots_dict, outfile)

with open(Val_txt, "w") as outfile:
    json.dump(val_roots_dict, outfile)

with open(Test_txt, "w") as outfile:
    json.dump(test_roots_dict, outfile)

