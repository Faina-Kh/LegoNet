import os
import shutil
import numpy as np
import csv
import cv2
import random
from os.path import exists
import shutil
import json
import pandas as pd




########################################################################################################################
current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

random.seed(0)
np.random.seed(0)

##############################################################################################################################################

# have the images, need to get corrected files

# define paths
folds_num = 5
images_num = 531
Train_num = 383
Val_num = 46
Test_num = 102

dataset_name = "" #"Tomato 2020"  #"Corn 2020" #"Melon 2019" #"Tomato 2020" #"Tomato 2019" #"Pepper 2021"  # #"Tomato 2020" # "Melon 2018"
data_path = os.path.join("D:/Faina/roots_project", "Dataset for root color model\\17_1_data\\For root color model\\all images")
                         #"Dataset for root color model\\Selected images from Brian_Grapevine","Rootfly_subfolders_Grapevine roots") #os.path.join("D:\\Faina\\roots_project", "manual_camera", "July_22", dataset_name)


output_path = os.path.join("E:\\roots_project", "Grapes_K_fold")
os.makedirs(output_path, exist_ok=True)

input_TRL = os.path.join(data_path, "TRL.csv")
input_points = os.path.join(data_path, "pointsOutput.csv")

txt_path = os.path.join(data_path, "all_data_Diameter_Length_Color.txt")

##########################################################################################################################

for i in range (folds_num):

    print("K= ", i+1)

    Train_dir = os.path.join(output_path,"K_"+str(i+1),"sub_Train"+"_"+str(i+1))
    Val_dir = os.path.join(output_path, "K_"+str(i+1), "sub_Val"+"_"+str(i+1))
    Test_dir = os.path.join(output_path,"K_"+str(i+1), "sub_Test"+"_"+str(i+1))

    os.makedirs(Train_dir, exist_ok=True)
    os.makedirs(Val_dir, exist_ok=True)
    os.makedirs(Test_dir, exist_ok=True)

    with_time_format = True
    save_cropped = False

    ####################################################################################################################

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

    ####################################################################################################################

    data = np.array(range(images_num))
    np.random.shuffle(data)
    print(data)

    input_TRL_df = pd.read_csv(input_TRL, header=None).to_numpy()
    Train_rows = input_TRL_df[data[:Train_num]]
    Val_rows = input_TRL_df[data[Train_num:(Train_num+Val_num)]]
    Test_rows = input_TRL_df[data[(Train_num+Val_num):]]

    Train_names=[]
    Val_names=[]
    Test_names=[]

    Train_P_rows = []
    Val_P_rows = []
    Test_P_rows = []

    train_roots_dict={}
    val_roots_dict={}
    test_roots_dict={}

    for row in Train_rows:
        img_name = row[0]
        img_path = os.path.join(data_path, img_name)
        image = cv2.imread(img_path)
        if save_cropped:
            image = image[21:471, 16:, :]
        dst_img_file = os.path.join(Train_dir, row[0])
        if save_cropped:
            cv2.imwrite(dst_img_file, image)
        else:
            shutil.copyfile(img_path, dst_img_file)
        Train_names.append(row[0])

        for row in Val_rows:
            img_name = row[0]
            img_path = os.path.join(data_path, img_name)
            image = cv2.imread(img_path)
            dst_img_file = os.path.join(Val_dir, row[0])
            if save_cropped:
                cv2.imwrite(dst_img_file, image)
            else:
                shutil.copyfile(img_path, dst_img_file)
            Val_names.append(row[0])

        for row in Test_rows:
            img_name = row[0]
            img_path = os.path.join(data_path, img_name)
            image = cv2.imread(img_path)
            dst_img_file = os.path.join(Test_dir, row[0])
            if save_cropped:
                cv2.imwrite(dst_img_file, image)
            else:
                shutil.copyfile(img_path, dst_img_file)
            Test_names.append(row[0])

    ####################################################################################################################

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

    ####################################################################################################################

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

    ####################################################################################################################

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

