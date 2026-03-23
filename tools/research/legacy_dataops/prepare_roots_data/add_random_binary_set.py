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

current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################

##############################################################################################################################################


images_dir = "D:\\Faina\\roots_project\\Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\original\\No root images from other auto cameras"

Train_dir = os.path.join(images_dir, "Train")
Val_dir = os.path.join(images_dir, "Val")
Test_dir = os.path.join(images_dir, "Test")

os.makedirs(Train_dir, exist_ok=True)
os.makedirs(Val_dir, exist_ok=True)
os.makedirs(Test_dir, exist_ok=True)

output_Train = os.path.join(images_dir, Train_dir, "more_Train.csv")
output_Val = os.path.join(images_dir, Val_dir, "more_Val.csv")
output_Test = os.path.join(images_dir, Test_dir, "more_Test.csv")

###############################################################################################################################################

Train_rows=[]
Val_rows=[]
Test_rows=[]


dir_list = os.listdir(images_dir)

for x in dir_list:
    # read images names
    if x.endswith(".jpg"):
        img_path = os.path.join(images_dir, x)
        rnd = np.random.random()
        # copy images
        if rnd <= 0.7:
            Train_rows.append([x,0])
            dst_img_file = os.path.join(Train_dir, x)
            shutil.copyfile(img_path, dst_img_file)

        elif 0.7 < rnd <= 0.8:
            Val_rows.append([x,0])
            dst_img_file = os.path.join(Val_dir, x)
            shutil.copyfile(img_path, dst_img_file)

        else:
            Test_rows.append([x,0])
            dst_img_file = os.path.join(Test_dir, x)
            shutil.copyfile(img_path, dst_img_file)


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
