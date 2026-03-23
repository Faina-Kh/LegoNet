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


dataset_name = "" #"Tomato 2020"  #"Corn 2020" #"Melon 2019" #"Tomato 2020" #"Tomato 2019" #"Pepper 2021"  # #"Tomato 2020" # "Melon 2018"
data_path = os.path.join("D:/Faina/roots_project", "Dataset for root color model\\17_1_data\\For root color model\\all images")
                         #"Dataset for root color model\\Selected images from Brian_Grapevine","Rootfly_subfolders_Grapevine roots") #os.path.join("D:\\Faina\\roots_project", "manual_camera", "July_22", dataset_name)


output_path = os.path.join("E:\\roots_project", "Grapes_K_fold")
os.makedirs(output_path, exist_ok=True)


input_TRL = os.path.join(data_path, "TRL_limit_5.csv")
input_points = os.path.join(data_path, "pointsOutput_limit_5.csv")
input_txt = os.path.join(data_path, "all_data_Diameter_Length_Color_limit_5.txt")


#create input dict
input_TRL_dict={}
input_points_dict={}
input_txt_dict={}
with open(input_TRL) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        input_TRL_dict[row[0]]=row

with open(input_points) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        input_points_dict[row[0]]=row

with open(input_txt) as f:
    data = f.read()

input_txt_dict = json.loads(data)

############################################################################################################################################

for i in range (folds_num):

    print("K= ", i+1)

    sets=['Train', 'Val', 'Test']

    for set in sets:
        current_dir = os.path.join(output_path,"K_"+str(i+1),"sub_"+ set +"_"+str(i+1))

        current_input_TRL = os.path.join(current_dir, set+".csv")
        current_im_names=[]


        # create output files
        out_TRL_file = os.path.join(current_dir, set+"_limit_5.csv")
        out_pointsOutput_file = os.path.join(current_dir, set+"_pointsOutput_limit_5.csv")
        out_txt = os.path.join(current_dir, set+"_Dia_Length_Color_limit_5.txt")


        with open(current_input_TRL) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                current_im_names.append(row[0])

        TRL_rows = []
        P_rows = []
        roots_dict={}
        with open(input_TRL) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
               if row[0] in current_im_names:
                   TRL_rows.append(row)

        with open(input_points) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] in current_im_names:
                    P_rows.append(row)

        for key in input_txt_dict.keys():
            img_name = input_txt_dict[key]['processed_name']
            if img_name in current_im_names:
                roots_dict[key]=input_txt_dict[key]

        ##############################################################################################################

        #print to files

        f_TRL = open(out_TRL_file, 'w', newline='')
        with f_TRL:
            writer = csv.writer(f_TRL)
            for row in TRL_rows:
                writer.writerow(row)

        f_points = open(out_pointsOutput_file, 'w', newline='')
        with f_points:
            writer = csv.writer(f_points)
            for row in P_rows:
                writer.writerow(row)

        with open(out_txt, "w") as outfile:
                json.dump(roots_dict, outfile)



