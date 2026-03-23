import os
import csv
import shutil
import json



# if have both original and processed images, and if need to use sub folders (Train, val, Test ) use the whole data points files,
# but use the images folder of the original data


data_path = os.path.join("D:\\Faina\\Roots\\Xu\\with_anns\\renamed_random_jpg_b") #"C:\\Users\\Aragorn\\Desktop\\roots project" #, "manual_camera", "July_22" )#"13_5_22\\Correction after visual inspection_20220513")

set_names = ["0h\\RGB roots", "6h\\RGB roots", "24h\\RGB roots", "72h\\RGB roots", "168h\\RGB roots"]
#["newAutoCam_17_7_22\\original_28_7_22", "July_22\\all_manual_2" ]
#, "Grapevine_data_all"]
#["Melon 2018", "Melon 2019", "Pepper 2021", "Tomato 2019", "Tomato 2020", "Corn 2020"]

target_dir = os.path.join(data_path, "RGB_Joined datasets") # "Joined datasets" #"Three_datasets_detection")


#############################################
generate_csv = True
generate_json = True
copy_images = True
############################################


for subset in ["Train", "Val", "Test"]:
    new_dict = {}
    rows1 = []
    rows2 = []

    target_txt = os.path.join(data_path, target_dir,  "sub_" + subset, subset+"_joined.txt")
    copy_to = os.path.join(data_path, target_dir,  "sub_" + subset)
    os.makedirs(copy_to, exist_ok=True)

    joined_TRL = os.path.join(copy_to, subset + ".csv")
    joined_points = os.path.join(copy_to, subset + "_pointsOutput.csv")

    for d_set in set_names:
        current_dir = os.path.join(data_path, d_set, "sub_" + subset)

        T1 = os.path.join(current_dir, subset + "_Dia_and_Length.txt")
        T2 = os.path.join(current_dir, subset + "_manual_Dia_and_Length.txt")
        T3 = os.path.join(current_dir, subset + "_Dia_Length_Color.txt")
        if os.path.exists(T1):
            input_txt =T1
        elif os.path.exists(T2):
            input_txt =T2
        elif os.path.exists(T3):
            input_txt =T3

        TRL_path = os.path.join(current_dir, subset + ".csv")
        points_path = os.path.join(current_dir, subset + "_pointsOutput.csv")

        dir_list = os.listdir(current_dir)

        if copy_images:
            # copy images
            for x in dir_list:
                # read images names
                if x.endswith(".jpg"):
                    shutil.copyfile(os.path.join(current_dir, x), os.path.join(copy_to, x))

        # copy from txt
        with open(input_txt) as f:
            data = f.read()

        roots_dict = json.loads(data)

        for key in roots_dict:
            new_dict[d_set+ "_"+key] = roots_dict[key]


        # copy TRL and points data to joined files
        count1=0
        count2=0
        with open(TRL_path, mode='r') as a1:
            reader1 = csv.reader(a1)
            for row in reader1:
                rows1.append(row)
                count1+=1

        with open(points_path, mode='r') as a2:
            reader2 = csv.reader(a2)
            for row in reader2:
                rows2.append(row)
                count2+=1

        print('dataset:', d_set)
        print("dict lenght:", len(roots_dict))
        print("TRL len:", count1)
        print("points len:", count2)


    # write to files
    if generate_json:
        with open(target_txt, "w") as outfile:
            json.dump(new_dict, outfile)

    if generate_csv:
        with open(joined_TRL, mode='w') as f1:
            writer1 = csv.writer(f1, lineterminator='\n')
            for row in rows1:
                writer1.writerow(row)


        with open(joined_points, mode='w') as f2:
            writer2 = csv.writer(f2, lineterminator='\n')
            for row in rows2:
                writer2.writerow(row)


############################################################################################################################
#
# img_path = os.path.join("D:/Faina/roots_project", "Rootfly_cam3", "June 22", "draw_procc_GT_June")
#
# copy_to = os.path.join("D:/Faina/roots_project", "Rootfly_cam3", "June 22", "draw_procc_GT_June_try")
# os.makedirs(copy_to, exist_ok=True)
#
#
# names_file = os.path.join("D:/Faina/roots_project", "Rootfly_cam3", "June 22", "current_try.csv")
#
# current_names = []
# with open(names_file, mode='r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         current_names.append(row[0])
#
#
# dir_list = os.listdir(img_path)
#
# for x in dir_list:
#     # read images names
#     if x.endswith(".jpg"):
#         if x.split("points_")[1] in current_names:
#             shutil.copyfile(os.path.join(img_path, x), os.path.join(copy_to, x))
#

###################################################################################################################################
#
# dataset_name = "Rootfly_cam3"  #"5_5_22\\Corrected annotation\\pepper_2021" #"corrected\\Tomato 2020"
# data_path = os.path.join("D:/Faina/roots_project", dataset_name)
#
# current_TRL = os.path.join(data_path, "sub_Train", "Train_June_try.csv")
# copy_from = os.path.join(data_path, "sub_Train", "Train_pointsOutput_June.csv")
#
# copy_to_path = os.path.join(data_path, "sub_Train", "Train_pointsOutput_June_try.csv")
#
# copy_to_list=[]
#
#
# current_names = []
# with open(current_TRL, mode='r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         current_names.append(row[0])
#
#
# with open(copy_from, mode='r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if row[0] in current_names:
#             copy_to_list.append(row)
#
#
# f = open(copy_to_path, 'w', newline='')
# with f:
#     writer = csv.writer(f)
#     for row in copy_to_list:
#         writer.writerow(row)





####################################################################################################################
# # if have both original and processed images, and if need to use sub folders (Train, val, Test ) use the whole data points files,
# # but use the images folder of the original data
#
# set_names = ["melon 2019", "Pepper 2021", "Tomato 2019"] #"Corn 2020"
#
# joined_path = os.path.join("D:\\Faina\\roots_project", "joined_data_partial_corrections")
#
# #
# # for d_set in set_names:
# #     data_path = os.path.join("D:\\Faina\\roots_project", "manual_camera", "partial_corrections", d_set)
# #
# #     for subset in ["Train", "Val", "Test"]:
# #         dir_list = os.listdir(os.path.join(data_path, "sub_"+subset))
# #         target = os.path.join(joined_path, "sub_"+subset)
# #         for x in dir_list:
# #             # read images names
# #             if x.endswith(".jpg"):
# #                 # copy images
# #                 shutil.copyfile(os.path.join(data_path, "sub_"+subset, x), os.path.join(target,x))
# #
# #
# #         # copy TRL and points data to joined files
# #         TRL_path = os.path.join(data_path, "sub_"+subset, subset+".csv")
# #         points_path = os.path.join(data_path, "sub_"+subset, subset+"_pointsOutput.csv")
# #         joined_TRL = os.path.join(joined_path,  "sub_"+subset, subset+".csv")
# #         joined_points = os.path.join(joined_path, "sub_" + subset, subset + "_pointsOutput.csv")
# #
# #         rows1=[]
# #         rows2=[]
# #
# #
# #         with open(TRL_path, mode='r') as a1:
# #             reader1 = csv.reader(a1)
# #             for row in reader1:
# #                 rows1.append(row)
# #
# #         with open(joined_TRL, mode='a') as f1:
# #             writer1 = csv.writer(f1, lineterminator='\n')
# #             for row in rows1:
# #                 writer1.writerow(row)
# #
# #         with open(points_path, mode='r') as a2:
# #             reader2 = csv.reader(a2)
# #             for row in reader2:
# #                 rows2.append(row)
# #
# #         with open(joined_points, mode='a') as f2:
# #             writer2 = csv.writer(f2, lineterminator='\n')
# #             for row in rows2:
# #                 writer2.writerow(row)
# #
#
# for subset in ["Train", "Val", "Test"]:
#     joined_points = os.path.join(joined_path, "sub_" + subset, subset + "_pointsOutput.csv")
#     joined_list = []
#     with open(joined_points, mode='r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             s = [row[i] for i in range(len(row)) if row[i]!=""]
#             joined_list.append(s)
#
#
#     joined_points_2 = os.path.join(joined_path, "sub_" + subset, subset + "_pointsOutput_2.csv")
#     f = open(joined_points_2, 'w', newline='')
#     with f:
#         writer = csv.writer(f)
#         for row in joined_list:
#             writer.writerow(row)