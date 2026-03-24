import os
import shutil
import numpy as np
import csv
import random
from random import sample

#################################################################################################################

current_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# paths to existing data

data_dir = os.path.join("D:/Faina/roots_project", "Kaining_tomato_2019/KNNETHOUSE_images")

relevant_img_file = os.path.join(data_dir, "TRL.csv")

num_images = 18410
#####################################################################################################################

# paths for new data

splitted_data_dir = os.path.join(data_dir, "splitted_data3")

processed_noRoot_dir = os.path.join(splitted_data_dir, "processed_noRoot")
processed_withRoot_dir = os.path.join(splitted_data_dir, "processed_withRoot")

# create folders if don't exist
os.makedirs(processed_noRoot_dir, exist_ok=True)
os.makedirs(processed_withRoot_dir, exist_ok=True)


# splitted files
processed_noRoot_file = os.path.join(processed_noRoot_dir, "processed_noRoot.csv")
processed_withRoot_file = os.path.join(processed_withRoot_dir, "processed_withRoot.csv")

processed_noRoot_rows = []
processed_withRoot_rows = []

#np.random.seed(0)

random.seed(0)

count= 0

withRoots_count = 0
noRoots_count = 0

# count images with and without roots
with open(relevant_img_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if float(row[1]) == 0:
            processed_noRoot_rows.append(row)
            noRoots_count +=1
        else:
            processed_withRoot_rows.append(row)
            withRoots_count +=1

print("withRoots_count = ",withRoots_count, ", ",  "noRoots_count = ", noRoots_count)
assert noRoots_count+ withRoots_count == num_images

# randomly choose 50 images from each category
processed_noRoot_rows = sample(processed_noRoot_rows, 50)
processed_withRoot_rows = sample(processed_withRoot_rows, 50)

# copy the chosen images to separate folders

for row in processed_noRoot_rows:
         dst_img_file_processesd = os.path.join(processed_noRoot_dir, row[0])
         shutil.copyfile(os.path.join(data_dir, row[0]), dst_img_file_processesd)

for row in processed_withRoot_rows:
    dst_img_file_processesd = os.path.join(processed_withRoot_dir, row[0])
    shutil.copyfile(os.path.join(data_dir, row[0]), dst_img_file_processesd)

# create the csv files

f_processed_noRoot = open(processed_noRoot_file, 'w', newline='')
with f_processed_noRoot:
    writer = csv.writer(f_processed_noRoot)
    for row in processed_noRoot_rows:
        writer.writerow(row)

f_processed_withRoot = open(processed_withRoot_file, 'w', newline='')
with f_processed_withRoot:
    writer = csv.writer(f_processed_withRoot)
    for row in processed_withRoot_rows:
        writer.writerow(row)
