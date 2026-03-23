import numpy as np
import csv
import random
from random import sample
import os
import shutil

##############################################################################################################################################

random.seed(0)
np.random.seed(0)

#################################################################################################################

current_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################

##############################################################################################################################################


images_dir = "D:\\Faina\\roots_project\\Rootfly_cam3\\Aug 22\\raw_images"
#"D:\\Faina\\roots_project\\Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\original"

input_file = os.path.join(images_dir, "autoCam3_test_TRL_raw.csv")
output_file = os.path.join(images_dir, "autoCam3_test_TRL_raw_binary.csv")

output_rows = []

with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        new_row=[]
        new_row.append(row[0])
        if float(row[1])>0:
            new_row.append(1)
        else:
            new_row.append(0)

        output_rows.append(new_row)


output1 = open(output_file, 'w', newline='')
with output1:
    writer = csv.writer(output1)
    for row in output_rows:
        writer.writerow(row)



########################################################################################################################

Train_dir = os.path.join(images_dir, "sub_Train")
Val_dir = os.path.join(images_dir, "sub_Val")
Test_dir = os.path.join(images_dir, "sub_Test")

os.makedirs(Train_dir, exist_ok=True)
os.makedirs(Val_dir, exist_ok=True)
os.makedirs(Test_dir, exist_ok=True)

input_Train = os.path.join(images_dir, Train_dir, "Train_pointsOutput_binary.csv")
input_Val = os.path.join(images_dir, Val_dir, "Val_pointsOutput_binary.csv")
input_Test = os.path.join(images_dir, Test_dir, "Test_pointsOutput_binary.csv")

output_Train = os.path.join(images_dir, Train_dir, "Train_pointsOutput_binary2.csv")
output_Val = os.path.join(images_dir, Val_dir, "Val_pointsOutput_binary2.csv")
output_Test = os.path.join(images_dir, Test_dir, "Test_pointsOutput_binary2.csv")


###############################################################################################################################################

Train_rows=[]
Val_rows=[]
Test_rows=[]

with open(input_Train) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        new_row=[]
        for x in row:
            if  x!="":
                new_row.append(x)
            else:
                break
        Train_rows.append(new_row)


with open(input_Val) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        new_row=[]
        for x in row:
            if  x!="":
                new_row.append(x)
            else:
                break
        Val_rows.append(new_row)

with open(input_Test) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        new_row = []
        for x in row:
            if x != "":
                new_row.append(x)
            else:
                break
        Test_rows.append(new_row)

output1 = open(output_Train, 'w', newline='')
with output1:
    writer = csv.writer(output1)
    for row in Train_rows:
        writer.writerow(row)


output2 = open(output_Val, 'w', newline='')
with output2:
    writer = csv.writer(output2)
    for row in Val_rows:
        writer.writerow(row)



output3 = open(output_Test, 'w', newline='')
with output3:
    writer = csv.writer(output3)
    for row in Test_rows:
        writer.writerow(row)
