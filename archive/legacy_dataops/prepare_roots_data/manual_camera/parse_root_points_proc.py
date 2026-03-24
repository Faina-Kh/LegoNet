import os
import csv
import json
import numpy as np


# if have both original and processed images, and if need to use sub folders (Train, val, Test ) use the whole data points files,
# but use the images folder of the original data

dataset_name = "" #Tube 30_not working" #"Tube 43" #"MELON 2018_tube 40" #"MELON 2018_tube 17" #"PEPPER 2021_tube 8" #"tomato 2020_tube 13" #"TOMATO 2019_tube 4" #"MELON 2019_tube 9" #"MELON 2018_tube 17" #"CORN 2020_tube 16"  #"Parthasarathi_tomato_Tube 9"
data_path = os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all", "Tube 22_renamed")
#("D:/Faina/roots_project", "Dataset for root color model", "17_1_data\\For root color model\\all images")#\\
                         # "Selected images from Brian_Grapevine","Rootfly_subfolders_Grapevine roots")
                         #"Chosen images for root diameter", dataset_name)
            #"manual_camera", "Sep 22", "NEW- manual camera", dataset_name) # "July_22" ,
            #os.path.join("D:\Faina\\roots_project", "Autocam image for CNN model", "Images for training\\Training dataset_automated camera")
            #os.path.join("D:\\Faina\\roots_project", "manual_camera", "13_5_22\\Correction after v"PEPPER 2021_tube 8"isual inspection_20220513", dataset_name) #"corrected"

#raw input data paths
# withRoot_images = os.path.join(data_path, "pepperm2021-splitted_data" , "processed_withRoot") #"tomato 2020- splitted_data3"
# withRoot_file = os.path.join(withRoot_images, "pepper_corrected_processed_withRoot.csv")
# noRoot_images = os.path.join(data_path, "pepperm2021-splitted_data" , "processed_noRoot")
# noRoot_file = os.path.join(noRoot_images, "processed_noRoot.csv")

getTube = False
Tube = "22" #"30" #""


relevant_img_file = os.path.join(data_path, "TRL.csv" ) #dataset_name+"_TRL.csv")
                                 #"original", "new_auto_cam_TRL_with_NoRoots.csv") #os.path.join(data_path, dataset_name+"_corrected.csv")

all_txt_paths = os.path.join(data_path, "points")
                #"Autocam3_files", "points data")
                #os.path.join(data_path, dataset_name+"-points") #dataset_name + "- points") #"-cordinates points")

root_data_file = os.path.join(data_path, "roots_info.txt") #"roots_info_with color.txt")
                #os.path.join(data_path, "roots_info.txt") #"new_auto_cam_roots_info.txt")
                #os.path.join(data_path, dataset_name+"_roots_info.txt")

# output path
pointsOutput_file = os.path.join(data_path, "pointsOutput.csv") #"pointsOutput_"+"new_autoCam"+".csv") #os.path.join(data_path, "pointsOutput.csv")
json_output_path = os.path.join(data_path, "all_data_Diameter_Length_Color.txt") #os.path.join(data_path, dataset_name+"_all_data_Diameter_and_Length.txt")


generate_csv = True
generate_json = True

from_orig_to_procc_points = False
H_ratio = 480 / 1944
W_ratio = 640 / 2592

########################################################################################################################

have_orig_images = False

with_time_format = True  # True or False

# is the data split to train/val/test
have_spllited_data = False

# move points coordinates for cropped images
move_points = False

########################################################################################################################

if have_spllited_data:
    if have_orig_images:
        images_match_file = os.path.join(data_path, "myFiles", "images_match.csv")

    # splitted input files
    Train_csv_file = os.path.join(data_path, "sub_Train", "Train.csv")
    Val_csv_file = os.path.join(data_path, "sub_Val", "Val.csv")
    Test_csv_file = os.path.join(data_path, "sub_Test", "Test.csv")

    # splitted output points files
    Train_pointsOutput_file = os.path.join(data_path, "sub_Train", "Train_pointsOutput.csv")
    Val_pointsOutput_file = os.path.join(data_path, "sub_Val", "Val_pointsOutput.csv")
    Test_pointsOutput_file = os.path.join(data_path, "sub_Test", "Test_pointsOutput.csv")

    Train_json_file = os.path.join(data_path, "sub_Train", "Train_Dia_and_Length.txt")
    Val_json_file = os.path.join(data_path, "sub_Val", "Val_Dia_and_Length.txt")
    Test_json_file = os.path.join(data_path, "sub_Test", "Test_Dia_and_Length.txt")

    train_img_names = []
    val_img_names = []
    test_img_names = []

    train_image_dict = {}
    val_image_dict = {}
    test_image_dict = {}

    train_roots_dict = {}
    val_roots_dict = {}
    test_roots_dict = {}


else:
    img_names = []
    image_dict = {}


if have_orig_images:
    relevant_name = "original_name"
else:
    relevant_name = "processed_name" #"proceesed_name"


##################################################################################################

# reading the data from the file
with open(root_data_file) as f:
    data = f.read()

roots_dict = json.loads(data)

################################################################################################

processed_to_orig = {}

if have_orig_images:
    with open(images_match_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            p = rows[0]
            o = rows[1].split("\\")[1]
            l = rows[2]

            processed_to_orig[p] = [o, l]

if have_spllited_data:
    with open(Train_csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            train_img_names.append((rows[0]))

    with open(Val_csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            val_img_names.append((rows[0]))

    with open(Test_csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            test_img_names.append((rows[0]))



# create a match between images name (based on T_L_S) and points data
image_dict = {}
for file in [relevant_img_file]: #[withRoot_file, noRoot_file]:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            name_split = row[0].split("_")
            if len(name_split) > 5:
                current_T = int(name_split[1].split("T")[1])
                current_L = name_split[2].split("L")[1]
            else:
                current_T = int(name_split[0].split("T")[1])
                current_L = name_split[1].split("L")[1]

            if "Brian_Grapevine" in data_path:
                if len(current_L) > 3:
                    Lsplit = [*current_L]
                    current_L = Lsplit[0]+Lsplit[1]+Lsplit[2]

            current_L = int(current_L)


            if with_time_format:
                if len(name_split) == 7:
                    current_Sess = int(name_split[5])
                elif len(name_split) == 5:
                    current_Sess = int(name_split[4].split(".jpg")[0])
            else:
                current_Sess = int(name_split[4].split('.jpg')[0])

            current_key = str(current_T)+ "_" + str(current_L)+ "_" + str(current_Sess)
            image_dict[current_key] = {}
            image_dict[current_key]["processed_name"]= row[0]

            if have_orig_images:
                image_dict[current_key]["original_name"] = processed_to_orig[row[0]][0]

            image_dict[current_key]["TRL"] = row[1]
            image_dict[current_key]["points"] = []
            image_dict[current_key]["roots_num"] = 0

            image_dict[current_key]["roots_dia_values"] = []
            image_dict[current_key]["roots_dia_mean"] = 0
            image_dict[current_key]["roots_dia_std"] = 0

            if have_spllited_data:
                if image_dict[current_key][relevant_name] in train_img_names:
                    train_image_dict[current_key] = {}
                    train_image_dict[current_key][relevant_name] = image_dict[current_key][relevant_name]
                    train_image_dict[current_key]["points"] = []

                    train_roots_dict[current_key] = image_dict[current_key]


                elif image_dict[current_key][relevant_name] in val_img_names:
                    val_image_dict[current_key] = {}
                    val_image_dict[current_key][relevant_name] = image_dict[current_key][relevant_name]
                    val_image_dict[current_key]["points"] = []

                    val_roots_dict[current_key] = image_dict[current_key]


                elif image_dict[current_key][relevant_name] in test_img_names:
                    test_image_dict[current_key] = {}
                    test_image_dict[current_key][relevant_name] = image_dict[current_key][relevant_name]
                    test_image_dict[current_key]["points"] = []

                    test_roots_dict[current_key] = image_dict[current_key]


lines = []
points_files_dir = os.listdir(all_txt_paths)

for x in points_files_dir:
    txt_path = os.path.join(all_txt_paths, x)
    print(x)
    #Tube = txt_path.split("_T")[1].split(".txt")[0]
    # Tube = txt_path.split("-T")[1].split(".txt")[0]
    if getTube:
        if dataset_name == "Parthasarathi_tomato_Tube 9":
            Tube = x.split("tube_")[1].split(".txt")[0]
        else:
            if "RAMATNEGEVWINES" in txt_path:
                Tube = x.split("RAMATNEGEVWINES")[0].split("T")[1].split("_")[0]
            if "-T" in x:
                 Tube = x.split("-T")[1].split(".txt")[0]
            if "_T" in x:
                if "_Tube" in x:
                    Tube = x.split("Tube")[1].split(".txt")[0]
                else:
                    Tube = x.split("Tube")[1].split(".txt")[0]
            elif "tube_" in x:
                 Tube = x.split("tube_")[1].split(".txt")[0]


            Tube = str(int(Tube))


    with open(txt_path) as f:
        lines = f.readlines()

        for line in lines:
            if ("Root" in line) and ("Root ID" not in line) and ("Roots in" not in line):
                RootID = int(line.split("):")[0].split("(")[2])

            if "LIVE" in line:
                session = int(line.split("Session")[1].split(":")[0])
                points = line.split("Session")[1].split("LIVE")[1].split("dia")[0]
                root_dia_info = line.split("Session")[1].split("LIVE")[1].split("dia")[1].split("=")[1].split("\n")[0]

                # the ratio between image Width in Rootfly and output width is 619/17.89~34.6
                # root of 0.4 dia is 7 in the points output file, meaning the output is R and not diameter

                # root_R = int(root_dia_info.split(";")[1].split(")")[0])
                root_Rx = int(root_dia_info.split(",")[0].split("(")[1])
                root_Ry = int(root_dia_info.split(",")[1])

                update_root_count = True
                splitted = points.split("(")
                for i in range(1,len(splitted)):
                    current = splitted[i].split(",")
                    Loc = int(current[2].split(")")[0])
                    image_dict_key = Tube + "_" + str(Loc) + "_" + str(session)

                    print("image", image_dict_key)

                    if image_dict_key=="11_110_1" or image_dict_key=="11_108_1":
                        continue

                    if image_dict_key in image_dict.keys():
                        if update_root_count:
                            image_dict[image_dict_key]["roots_num"] += 1
                            current_roots_num = image_dict[image_dict_key]["roots_num"]

                            image_dict[image_dict_key]["root_" + str(RootID)] = {}

                            image_dict[image_dict_key]["root_" + str(RootID)]["Root_Diameter"] = roots_dict[image_dict_key]["RootID_" + str(RootID)]["Diameter"]

                            #print(image_dict_key, "root", RootID)

                            image_dict[image_dict_key]["root_" + str(RootID)]["Root_Length"] = roots_dict[image_dict_key]["RootID_" + str(RootID)]["Length"]

                            image_dict[image_dict_key]["roots_dia_values"].append(roots_dict[image_dict_key]["RootID_" + str(RootID)]["Diameter"])

                            image_dict[image_dict_key]["root_" + str(RootID)]["Root_Color"] = roots_dict[image_dict_key]["RootID_" + str(RootID)]["Color"]

                            #image_dict[image_dict_key]["root_" + str(current_roots_num)]["root_dia"] = root_dia_info
                            if move_points:
                                root_Rx = root_Rx - 16
                                root_Ry = root_Ry - 21

                            if from_orig_to_procc_points:
                                root_Rx = int(root_Rx * W_ratio)
                                root_Ry = int(root_Ry * H_ratio)

                            image_dict[image_dict_key]["root_" + str(RootID)]["root_Rx"] = root_Rx
                            image_dict[image_dict_key]["root_" + str(RootID)]["root_Ry"] = root_Ry

                            image_dict[image_dict_key]["root_" + str(RootID)]["points"] = []

                            update_root_count = False

                        x = int(current[0])
                        y = int(current[1])

                        if move_points:
                            x = x - 16
                            y = y - 21

                        if from_orig_to_procc_points:
                            x = int(x * W_ratio)
                            y = int(y * H_ratio)

                        image_dict[image_dict_key]["points"].append(x)
                        image_dict[image_dict_key]["points"].append(y)

                        image_dict[image_dict_key]["root_" + str(RootID)]["points"].append(x)
                        image_dict[image_dict_key]["root_" + str(RootID)]["points"].append(y)

                        if have_spllited_data:
                            if image_dict_key in train_image_dict:
                                train_image_dict[image_dict_key]["points"].append(x)
                                train_image_dict[image_dict_key]["points"].append(y)

                                train_roots_dict[image_dict_key] = image_dict[image_dict_key]

                            elif image_dict_key in val_image_dict:
                                val_image_dict[image_dict_key]["points"].append(x)
                                val_image_dict[image_dict_key]["points"].append(y)

                                val_roots_dict[image_dict_key] = image_dict[image_dict_key]

                            elif image_dict_key in test_image_dict:
                                test_image_dict[image_dict_key]["points"].append(x)
                                test_image_dict[image_dict_key]["points"].append(y)

                                test_roots_dict[image_dict_key] = image_dict[image_dict_key]



#####################################################################################################################

# add avg and std of diameter per image

for key in image_dict.keys():
    if len(image_dict[key]["roots_dia_values"]) > 0:
        image_dict[key]["roots_dia_mean"] = np.mean(image_dict[key]["roots_dia_values"])
    if image_dict[key]["roots_num"] > 1:
        image_dict[key]["roots_dia_std"] = np.std(image_dict[key]["roots_dia_values"])

if have_spllited_data:
    for key in train_roots_dict.keys():
        if len(train_roots_dict[key]["roots_dia_values"]) > 0:
            train_roots_dict[key]["roots_dia_mean"] = np.mean(train_roots_dict[key]["roots_dia_values"])
        if train_roots_dict[key]["roots_num"] > 1:
            train_roots_dict[key]["roots_dia_std"] = np.std(train_roots_dict[key]["roots_dia_values"])

    for key in val_roots_dict.keys():
        if len(val_roots_dict[key]["roots_dia_values"]) > 0:
            val_roots_dict[key]["roots_dia_mean"] = np.mean(val_roots_dict[key]["roots_dia_values"])
        if val_roots_dict[key]["roots_num"] > 1:
            val_roots_dict[key]["roots_dia_std"] = np.std(val_roots_dict[key]["roots_dia_values"])

    for key in test_roots_dict.keys():
        if len(test_roots_dict[key]["roots_dia_values"]) > 0:
            test_roots_dict[key]["roots_dia_mean"] = np.mean(test_roots_dict[key]["roots_dia_values"])
        if test_roots_dict[key]["roots_num"] > 1:
            test_roots_dict[key]["roots_dia_std"] = np.std(test_roots_dict[key]["roots_dia_values"])


###################################################################################################################

if generate_json:

    # print data to json
    with open(json_output_path, "w") as outfile:
        json.dump(image_dict, outfile)

    if have_spllited_data:
        with open(Train_json_file, "w") as outfile:
            json.dump(train_roots_dict, outfile)

        with open(Val_json_file, "w") as outfile:
            json.dump(val_roots_dict, outfile)

        with open(Test_json_file, "w") as outfile:
            json.dump(test_roots_dict, outfile)



if generate_csv:

    # a file with all the data, not splitted
    f = open(pointsOutput_file, 'w', newline='')
    with f:
        writer = csv.writer(f)
        for im in image_dict.keys():
            #create the row for the output file
            myrow = []
            myrow.append(image_dict[im][relevant_name])
            current_points = image_dict[im]["points"]
            for i in current_points:
                myrow.append(i)

            writer.writerow(myrow)


    if have_spllited_data:
        f = open(Train_pointsOutput_file, 'w', newline='')
        with f:
            writer = csv.writer(f)
            for im in train_image_dict.keys():
                #create the row for the output file
                myrow = []
                myrow.append(train_image_dict[im][relevant_name])
                current_points = train_image_dict[im]["points"]
                for i in current_points:
                    myrow.append(i)

                writer.writerow(myrow)

        f = open(Val_pointsOutput_file, 'w', newline='')
        with f:
            writer = csv.writer(f)
            for im in val_image_dict.keys():
                #create the row for the output file
                myrow = []
                myrow.append(val_image_dict[im][relevant_name])
                current_points = val_image_dict[im]["points"]
                for i in current_points:
                    myrow.append(i)

                writer.writerow(myrow)

        f = open(Test_pointsOutput_file, 'w', newline='')
        with f:
            writer = csv.writer(f)
            for im in test_image_dict.keys():
                #create the row for the output file
                myrow = []
                myrow.append(test_image_dict[im][relevant_name])
                current_points = test_image_dict[im]["points"]
                for i in current_points:
                    myrow.append(i)

                writer.writerow(myrow)
