import os
import numpy as np
import csv
import cv2
import random
import shutil
import json
import pandas as pd


###############################################################################################################
random.seed(0)
np.random.seed(0)
###############################################################################################################

images_num = 288 #266 #821

Train_num = int(0.7*images_num)
Val_num = int(0.1*images_num)   # wrong :   int(0.2*images_num)
Test_num = images_num-Train_num-Val_num


#type = "GFP"
data_path = os.path.join("D:\\Faina", "Roots", "Sharon\\Hatzeva_all_images_daily","annotations")
#"D:\\Faina\\Roots\\Xu\\with_anns\\renamed_random_jpg_b", "all_images_with_anns", type) #"D:\\Faina\\Roots", "Sharon", "Hatzeva 2024-2025-Faina" #("C:\\Users\\Hydrus\\Desktop\\Faina\\Roots", "Sharon\\Hatzeva 2024-2025-Faina")
output_path = os.path.join(data_path, "splitted")
os.makedirs(output_path, exist_ok=True)

# input paths
input_TRL = os.path.join(data_path, "combined_TRL.csv") #type+"_TRL.csv"
input_points = os.path.join(data_path, "combined_pointsOutput.csv") #type+"_pointsOutput.csv"
txt_path = "" #os.path.join(data_path, "data_Diameter_and_Length.txt")


#output paths
Train_dir = os.path.join(output_path,"sub_Train")
Val_dir = os.path.join(output_path, "sub_Val")
Test_dir = os.path.join(output_path, "sub_Test")

os.makedirs(Train_dir, exist_ok=True)
os.makedirs(Val_dir, exist_ok=True)
os.makedirs(Test_dir, exist_ok=True)


# create spllitted files
Train_csv_file = os.path.join(Train_dir, "Train_TRL.csv")
Val_csv_file = os.path.join(Val_dir, "Val_TRL.csv")
Test_csv_file = os.path.join(Test_dir, "Test_TRL.csv")

# splitted output points files
Train_pointsOutput_file = os.path.join(Train_dir, "Train_pointsOutput.csv")
Val_pointsOutput_file = os.path.join(Val_dir, "Val_pointsOutput.csv")
Test_pointsOutput_file = os.path.join(Test_dir, "Test_pointsOutput.csv")


if txt_path!="":
    Train_txt = os.path.join(Train_dir, "Train_Dia_Length_Color.txt")
    Val_txt = os.path.join(Val_dir, "Val_Dia_Length_Color.txt")
    Test_txt = os.path.join(Test_dir, "Test_Dia_Length_Color.txt")


###############################################################################################################

data = np.array(range(images_num))
np.random.shuffle(data)
#print(data)

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

###############################################################################################################
print("Train images:")
for row in Train_rows:
    img_name = row[0]
    print(img_name)
    img_path = os.path.join(data_path, img_name)
    image = cv2.imread(img_path)

    dst_img_file = os.path.join(Train_dir, row[0])
    shutil.copyfile(img_path, dst_img_file)
    Train_names.append(row[0])

print("Val images:")
for row in Val_rows:
    img_name = row[0]
    print(img_name)
    img_path = os.path.join(data_path, img_name)
    image = cv2.imread(img_path)
    dst_img_file = os.path.join(Val_dir, row[0])
    shutil.copyfile(img_path, dst_img_file)
    Val_names.append(row[0])

print("Test images:")
for row in Test_rows:
    print(img_name)
    img_name = row[0]
    img_path = os.path.join(data_path, img_name)
    image = cv2.imread(img_path)
    dst_img_file = os.path.join(Test_dir, row[0])
    shutil.copyfile(img_path, dst_img_file)
    Test_names.append(row[0])

####################################################################################################################

with open(input_points) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = row[0]
        current_row = []
        for i in range(len(row)):
            if row[i]!="":
                current_row.append(row[i])
        if img_name in Train_names:
            Train_P_rows.append(current_row)
        elif img_name in Val_names:
            Val_P_rows.append(current_row)
        elif img_name in Test_names:
            Test_P_rows.append(current_row)

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

if txt_path!="":

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

