import os
import shutil
import numpy as np
import csv
from os.path import exists

#################################################################################################################

#current_gpu = '0'

#os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
#print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# paths to existing data

data_dir = os.path.join("D:\Faina\\roots_project", "Rootfly_cam3\\Aug 22")
                         #"Autocam image for CNN model", "Images for training\\Training dataset_automated camera")
            #os.path.join("D:\\Faina\\roots_project", "Rootfly_cam3") # "../data/"+ dataset_name
original_img_path = os.path.join(data_dir, "raw_images") #"cam3_original images")
processesd_img_path = os.path.join("D:\\Faina\\roots_project", "Rootfly_cam3", "processed_img")  #"processed_img")

relevant_img_file = os.path.join(data_dir, "autoCam3_test_TRL.csv")  #"June 22", "TRL_June.csv") #"myFiles\\TRL_cam3.csv")


num_images = 832
new_data = True

if new_data:
    sess_dict = {}
    to_split = False

    if to_split:

        input_Train_dir = os.path.join(data_dir, "Training dataset_Rootfly", "sub_Train")
        input_Val_dir = os.path.join(data_dir, "Training dataset_Rootfly", "sub_Val")
        input_Test_dir = os.path.join(data_dir, "Training dataset_Rootfly", "sub_Test")

        # outputs

        Train_dir = os.path.join(data_dir, "original", "sub_Train")
        Val_dir = os.path.join(data_dir, "original", "sub_Val")
        Test_dir = os.path.join(data_dir, "original", "sub_Test")

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



else:
    sess_dict = {"1": "006", "2": "011", "3": "016", "4": "021", "5": "026", "6": "031", "7": "036", "8": "041",
                 "9": "046", "10": "051", "11": "056", "12": "061", "13": "067", "14": "074", "15": "079", "16": "084",
                 "17": "089", "18": "094", "19": "099", "20": "104", "21": "109", "22": "114", "23": "119", "24": "124",
                 "25": "129", "26": "134"}

#####################################################################################################################

# paths for new data
images_match_file = os.path.join(data_dir, "images_match_autoCam3_test.csv")
images_match_rows = []

count= 0


if new_data:

    resized_dict = {}

    with open(relevant_img_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            resized_dict[row[0]] = row[1]

    dir_list = os.listdir(processesd_img_path)
    for dir_num in dir_list:
        resized_img_list = os.listdir(os.path.join(processesd_img_path, dir_num))
        for im in resized_img_list:
            if im.endswith('.jpg'):
                img_TRL = resized_dict[im]
                Loc = int(im.split("_")[2].split("L")[1])

                raw_img_list = os.listdir(os.path.join(original_img_path, dir_num))
                check = 0
                for raw_img in raw_img_list:
                    current_img_num = int(raw_img.split("_")[0].split("img")[1])
                    if current_img_num == 1:
                        check = 1
                        img_num = 32 - Loc + 1
                        break

                if check == 0:
                    # first img is img002
                    img_num = 32 - Loc + 2


                for x in raw_img_list:

                    current_img_num = int(x.split("_")[0].split("img")[1])

                    if current_img_num == img_num:
                        count += 1

                        images_match_rows.append([im, os.path.join(dir_num, x), resized_dict[im]])

                        if to_split:
                            print("copy file", count)

                            if exists(os.path.join(input_Train_dir, im)):
                                dst_img_file = os.path.join(Train_dir, x)
                                shutil.copyfile(os.path.join(original_img_path, dir_num, x), dst_img_file)
                                Train_rows.append([x, resized_dict[im]])

                            if exists(os.path.join(input_Val_dir, im)):
                                dst_img_file = os.path.join(Val_dir, x)
                                shutil.copyfile(os.path.join(original_img_path, dir_num, x), dst_img_file)
                                Val_rows.append([x, resized_dict[im]])

                            if exists(os.path.join(input_Test_dir, im)):
                                dst_img_file = os.path.join(Test_dir, x)
                                shutil.copyfile(os.path.join(original_img_path, dir_num, x), dst_img_file)
                                Test_rows.append([x, resized_dict[im]])

                        break

else:

    with open(relevant_img_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sess = int(row[0].split("_")[4].split(".jpg")[0])
            Loc = int(row[0].split("_")[2].split("L")[1])

            dir_num = sess_dict[str(sess)]

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

                     break


# create the csv files

f_match_images = open(images_match_file, 'w', newline='')
with f_match_images:
    writer = csv.writer(f_match_images)
    for row in images_match_rows:
        writer.writerow(row)


if to_split:

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
