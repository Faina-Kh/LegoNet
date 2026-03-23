import os
import shutil
import numpy as np
import csv
import random

#################################################################################################################

# current_gpu = '0'
#
# os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
# print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# paths to existing data

data_dir = os.path.join("D:\\Faina\\roots_project", "Rootfly_cam3") # "../data/"+ dataset_name
original_img_path = os.path.join(data_dir, "cam3_original images")
processesd_img_path = os.path.join(data_dir, "processed_img")

relevant_img_file = os.path.join(data_dir, "myFiles\\TRL_cam3.csv")


sess_dict = {"1": "006", "2": "011", "3": "016", "4": "021", "5": "026", "6": "031", "7": "036", "8": "041",
             "9": "046", "10": "051", "11": "056", "12": "061", "13": "067", "14": "074", "15": "079", "16": "084",
             "17": "089", "18": "094", "19": "099", "20": "104", "21": "109", "22": "114", "23": "119", "24": "124",
             "25": "129", "26": "134"}

num_images = 832

np.random.seed(0)
random.seed(0)
#####################################################################################################################

# paths for new data

splitted_data_dir = os.path.join(data_dir, "splitted_data")

processed_noRoot_dir = os.path.join(splitted_data_dir, "processed_noRoot")
processed_withRoot_dir = os.path.join(splitted_data_dir, "processed_withRoot")

original_noRoot_dir = os.path.join(splitted_data_dir, "original_noRoot")
original_withRoot_dir = os.path.join(splitted_data_dir, "original_withRoot")

# create folders if don't exist
os.makedirs(processed_noRoot_dir, exist_ok=True)
os.makedirs(processed_withRoot_dir, exist_ok=True)
os.makedirs(original_noRoot_dir, exist_ok=True)
os.makedirs(original_withRoot_dir, exist_ok=True)

# splitted files
processed_noRoot_file = os.path.join(processed_noRoot_dir, "processed_noRoot.csv")
processed_withRoot_file = os.path.join(processed_withRoot_dir, "processed_withRoot.csv")
original_noRoot_file = os.path.join(original_noRoot_dir, "original_noRoot.csv")
original_withRoot_file = os.path.join(original_withRoot_dir, "original_withRoot.csv")

images_match_file = os.path.join(splitted_data_dir, "images_match.csv")

processed_noRoot_rows = []
processed_withRoot_rows = []
original_noRoot_rows = []
original_withRoot_rows = []
images_match_rows = []

count= 0

withRoots_count = 0
noRoots_count = 0

# count images with and without roots
with open(relevant_img_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if float(row[1]) == 0:
            noRoots_count +=1
        else:
            withRoots_count +=1

print("withRoots_count = ",withRoots_count, "noRoots_count = ", noRoots_count)
assert noRoots_count+ withRoots_count == num_images

withRoots_chance = 50/withRoots_count
noRoots_chance = 50/noRoots_count


with open(relevant_img_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
         sess = int(row[0].split("_")[4].split(".jpg")[0])
         dir_num = sess_dict[str(sess)]
         Loc = int(row[0].split("_")[2].split("L")[1])

         current_dir = os.path.join(original_img_path, dir_num)
         dir_list = os.listdir(current_dir)
         check = 0
         for im in dir_list:
             current_img_num = int(im.split("_")[0].split("img")[1])
             if current_img_num==1:
                 check = 1
                 img_num = 32 - Loc + 1
                 break

         if check == 0:
             #first img is img002
             img_num = 32 - Loc + 2


         for x in dir_list:
             current_img_num = int(x.split("_")[0].split("img")[1])

             if current_img_num==img_num:
                 count += 1
                 print("copy file", count, "out of", num_images)

                 images_match_rows.append([row[0], os.path.join(dir_num, x), row[1]])

                 # choose randomly were to assign the image
                 rnd = np.random.random()

                 if float(row[1]) == 0:
                     #if the relevant image has no roots, randomly decide if to copy the image (processed anf original), based on noRoots_chance
                     if rnd <= noRoots_chance:
                         original_noRoot_rows.append([x, row[1]])
                         processed_noRoot_rows.append([row[0], row[1]])

                         # copy original image to original_noRoot_dir
                         dst_img_file_original = os.path.join(original_noRoot_dir, x)
                         shutil.copyfile(os.path.join(current_dir, x), dst_img_file_original)

                         # copy the matching processed image to original_noRoot_dir
                         dst_img_file_processesd = os.path.join(processed_noRoot_dir, row[0])
                         shutil.copyfile(os.path.join(processesd_img_path, row[0]), dst_img_file_processesd)
                 else:
                     # for image with roots, randomly decide if to copy the image (processed anf original), based on withRoots_chance
                     if rnd <= withRoots_chance:
                        original_withRoot_rows.append([x, row[1]])
                        processed_withRoot_rows.append([row[0], row[1]])

                        # copy original image to original_withRoot_dir
                        dst_img_file_original = os.path.join(original_withRoot_dir, x)
                        shutil.copyfile(os.path.join(current_dir, x), dst_img_file_original)

                        # copy the matching processed image to original_withRoot_dir
                        dst_img_file_processesd = os.path.join(processed_withRoot_dir, row[0])
                        shutil.copyfile(os.path.join(processesd_img_path, row[0]), dst_img_file_processesd)


                 break


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

f_original_noRoot = open(original_noRoot_file, 'w', newline='')
with f_original_noRoot:
    writer = csv.writer(f_original_noRoot)
    for row in original_noRoot_rows:
        writer.writerow(row)

f_original_withRoot = open(original_withRoot_file, 'w', newline='')
with f_original_withRoot:
    writer = csv.writer(f_original_withRoot)
    for row in original_withRoot_rows:
        writer.writerow(row)



f_match_images = open(images_match_file, 'w', newline='')
with f_match_images:
    writer = csv.writer(f_match_images)
    for row in images_match_rows:
        writer.writerow(row)
