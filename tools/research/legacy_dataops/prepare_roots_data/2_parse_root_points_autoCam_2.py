import os
import csv
import json
import numpy as np

#################################################################################################################

# current_gpu = '0'
#
# os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
# print('Running on gpu {}'.format(current_gpu))

############################################################################################################################
# if have both original and processed images, and if need to use sub folders (Train, val, Test ) use the whole data points files,
# but use the images folder


#type = "RGB" #"GFP" #"RGB"
#time="168h"

data_path = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names",
                          "daily_by_cam", "cam9\\Tube_9_wrong sessions","additional_annotations")
#("D:\\Faina\\Roots", "Xu\\with_anns\\renamed_random_jpg_b", time, type+" roots")
                         #"Sharon", "Hatzeva 2024-2025-Faina" #"D:\\Faina\\roots_project", "Sharon auto")
                         #"raw_images", "cam"+Tube+"-T00"+Tube) #"Rootfly_cam3")
                         #"Autocam image for CNN model", "Images for training\\Training dataset_automated camera")
#os.path.join("D:\\Faina\\roots_project", "Rootfly_cam3")
current_tube_path = "" #os.path.join(data_path, "raw_images\\cam"+Tube+"-T00"+Tube+"\\Cam "+Tube+ " data")

# input files
Tube = "9" #"6" #"1" ""
relevant_img_file = os.path.join(data_path, "cam9_Tube_9_TRL.csv") #"all_TRL_processed.csv") #"Aug 22", "autoCam3_test_TRL.csv") #"June 22", "TRL_June.csv") #"TRL_cam3.csv")
root_data_file = os.path.join(data_path, "cam9_Tube_9_roots_info.txt") #"all_roots_info.txt")#"Aug 22", "autoCam3_test_roots_info.txt")  #"June 22", "roots_Diameter_and_Length_June.txt")

all_txt_paths = "" #os.path.join(data_path, "Coordinates") #current_tube_path, "points") #"Aug 22", "points data") #"Autocam3_files", "points data")

# Rootflys txt
txt_path = os.path.join(data_path, "Cam9- Tube 9- session 9-New.txt") #"rootfly_logfile.txt") #"Training dataset_Rootfly", "Autocam3_files")
                        #"June 22", "autocam_point data_20220617.txt")  #"myFiles", "cam3_Points.txt")

# output files
pointsOutput_file = os.path.join(data_path, "cam9_Tube_9_pointsOutput.csv") #"Training dataset_Rootfly", "pointsOutput_"+"new_autoCam"+".csv") #"June 22", "pointsOutput_June.csv")
json_output_path = os.path.join(data_path, "cam9_Tube_9_data_Diameter_and_Length.txt")
                    #"Training dataset_Rootfly", "all_data_Diameter_and_Length.txt")
                   #os.path.join(data_path, "June 22", "all_data_Diameter_and_Length_June_new.txt")

generate_csv = True
generate_json = True

have_orig_images = True
original_names = False
# if the raw data was generated on the processed images - it needs to be resized for the size of auto cam
resized_for_Rootfly = True

images_match_file = "" #os.path.join(current_tube_path, "images_match_"+Tube+".csv") #"images_match_1_6.csv") #"images_match_"+Tube+".csv") #"myFiles",

with_time_format = False  # True or False

orig_H = 1944
orig_W = 2592

if "Xu" in data_path or not resized_for_Rootfly:
    H_ratio = 1
    W_ratio = 1
elif resized_for_Rootfly:
    H_ratio = 1944 / 480
    W_ratio = 2592 / 640

#diagonal_ratio = np.sqrt(np.square(W_ratio)+ np.square(H_ratio))



have_spllited_data = False
if have_spllited_data:

    # splitted input files
    Train_dir = os.path.join(data_path, "sub_Train")
    Val_dir = os.path.join(data_path, "sub_Val")
    Test_dir = os.path.join(data_path, "sub_Test")

    Train_csv_file = os.path.join(Train_dir, "Train.csv")
    Val_csv_file = os.path.join(Val_dir, "Val.csv")
    Test_csv_file = os.path.join(Test_dir, "Test.csv")

    # splitted output points files
    Train_pointsOutput_file = os.path.join(Train_dir, "Train_pointsOutput.csv")
    Val_pointsOutput_file = os.path.join(Val_dir, "Val_pointsOutput.csv")
    Test_pointsOutput_file = os.path.join(Test_dir, "Test_pointsOutput.csv")

    Train_json_file = os.path.join(Train_dir, "Train_Dia_and_Length.txt")
    Val_json_file = os.path.join(Val_dir, "Val_Dia_and_Length.txt")
    Test_json_file = os.path.join(Test_dir, "Test_Dia_and_Length.txt")


    train_img_names = []
    val_img_names = []
    test_img_names = []

    train_image_dict = {}
    val_image_dict = {}
    test_image_dict = {}

    train_roots_dict = {}
    val_roots_dict = {}
    test_roots_dict = {}




##################################################################################################

if have_orig_images and original_names:
    relevant_name = "original_name"
else:
    relevant_name = "processed_name" #"proceesed_name"


# reading the data from the file
with open(root_data_file) as f:
    data = f.read()

roots_dict = json.loads(data)

################################################################################################


processed_to_orig = {}
orig_to_procc = {}

if have_orig_images and original_names:
    #orig_to_procc = {}
    i=0
    with open(images_match_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            p = rows[0]
            o = rows[1].split("\\")[1]
            l = rows[2]
            processed_to_orig[p] = [o, l]

            orig_to_procc[o] = [p, l]

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
with open(relevant_img_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        name_split = row[0].split("_")

        if "L_" in row[0]:
            name_split = [name_split[0], name_split[1], name_split[2] + name_split[3], name_split[4], name_split[5]]

        if "Xu" in data_path:
            current_T = 1
            current_L = int(name_split[1].split("L")[1].split(".jpg")[0])
            current_Sess = 1
        else:
            current_T = int(name_split[1].split("T")[1])
            current_L = int(name_split[2].split("L")[1])

            if with_time_format:
                current_Sess = int(name_split[5])
            else:
                current_Sess = int(name_split[4].split('.jpg')[0])

        current_key = str(current_T)+ "_" + str(current_L)+ "_" + str(current_Sess)

        if have_orig_images and original_names:
            if row[0] in processed_to_orig.keys():
                image_dict[current_key] = {}
                image_dict[current_key]["processed_name"] = row[0]
                image_dict[current_key]["original_name"] = processed_to_orig[row[0]][0]
                del orig_to_procc[image_dict[current_key]["original_name"]]
            else:
                print(row[0])
                continue

        else:
            image_dict[current_key] = {}
            image_dict[current_key]["processed_name"] = row[0]

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


if len(orig_to_procc) > 0:
    # add additional no roots images without relevant resized image
    i=1
    for key in orig_to_procc.keys():
        current_key = "no_roots_" + str(i)
        image_dict[current_key] = {}
        image_dict[current_key]["processed_name"] = ""
        image_dict[current_key]["original_name"] = key
        image_dict[current_key]["TRL"] = 0
        image_dict[current_key]["points"] = []
        image_dict[current_key]["roots_num"] = 0

        image_dict[current_key]["roots_dia_values"] = []
        image_dict[current_key]["roots_dia_mean"] = 0
        image_dict[current_key]["roots_dia_std"] = 0

        if os.path.exists(os.path.join(Train_dir, key)):
            train_image_dict[current_key] = {}
            train_image_dict[current_key][relevant_name] = key
            train_image_dict[current_key]["points"] = []
            train_roots_dict[current_key] = image_dict[current_key]

        elif os.path.exists(os.path.join(Val_dir, key)):
            val_image_dict[current_key] = {}
            val_image_dict[current_key][relevant_name] = key
            val_image_dict[current_key]["points"] = []

            val_roots_dict[current_key] = image_dict[current_key]

        elif os.path.exists(os.path.join(Test_dir, key)):
            test_image_dict[current_key] = {}
            test_image_dict[current_key][relevant_name] = key
            test_image_dict[current_key]["points"] = []

            test_roots_dict[current_key] = image_dict[current_key]

        i+=1

lines = []

########################################################################################################

def parse_txt(image_dict,txt_path, check_tube, Tube="", train_image_dict=[], val_image_dict=[], test_image_dict=[]):
    if check_tube:
        if "_T" in txt_path:
            Tube = str(int(txt_path.split("_T")[1].split(".txt")[0]))
        elif "-T" in txt_path:
            Tube = str(int(txt_path.split("-T")[1].split(".txt")[0]))
        elif "Tube-" in txt_path:
            Tube = str(int(txt_path.split("Tube-")[1].split(".txt")[0]))
        elif "Tube " in txt_path:
            Tube = str(int(txt_path.split("Tube ")[1].split(".txt")[0]))
        else:
            assert Tube != ""

    print(txt_path, ", Tube = ", Tube)

    with (open(txt_path) as f):
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
                for i in range(1, len(splitted)):
                    current = splitted[i].split(",")
                    Loc = int(current[2].split(")")[0])

                    image_dict_key = Tube + "_" + str(Loc) + "_" + str(session)
                    print(image_dict_key)

                    if image_dict_key not in image_dict.keys():
                        continue

                    # if image_dict_key=="9_1_23":
                    #     a=1

                    if update_root_count:
                        image_dict[image_dict_key]["roots_num"] += 1
                        current_roots_num = image_dict[image_dict_key]["roots_num"]

                        image_dict[image_dict_key]["root_" + str(RootID)] = {}

                        if "Diameter" in roots_dict[image_dict_key]["RootID_" + str(RootID)].keys():
                            image_dict[image_dict_key]["root_" + str(RootID)]["Root_Diameter"] = \
                            roots_dict[image_dict_key]["RootID_" + str(RootID)]["Diameter"]

                            image_dict[image_dict_key]["roots_dia_values"].append(
                                roots_dict[image_dict_key]["RootID_" + str(RootID)]["Diameter"])

                        image_dict[image_dict_key]["root_" + str(RootID)]["Root_Length"] = \
                        roots_dict[image_dict_key]["RootID_" + str(RootID)]["Length"]


                        # image_dict[image_dict_key]["root_" + str(RootID)]["root_dia_info"] = root_dia_info
                        # image_dict[image_dict_key]["root_" + str(RootID)]["root_R"] = int(root_R*diagonal_ratio)
                        current_Rx = int(root_Rx * W_ratio)
                        if current_Rx < 0:
                            current_Rx = 0
                        if current_Rx > (orig_W - 1):
                            current_Rx = orig_W - 1
                        image_dict[image_dict_key]["root_" + str(RootID)]["root_Rx"] = current_Rx

                        current_Ry = int(root_Ry * H_ratio)
                        if current_Ry < 0:
                            current_Ry = 0
                        if current_Ry > (orig_H - 1):
                            current_Ry = orig_H - 1
                        image_dict[image_dict_key]["root_" + str(RootID)]["root_Ry"] = current_Ry

                        image_dict[image_dict_key]["root_" + str(RootID)]["points"] = []
                        update_root_count = False

                    x = int(current[0])
                    y = int(current[1])

                    x = int(x * W_ratio)
                    y = int(y * H_ratio)

                    if x < 0:
                        x = 0

                    if x > (orig_W - 1):
                        x = orig_W - 1

                    if y < 0:
                        y = 0

                    if y > (orig_H - 1):
                        y = orig_H - 1

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


    if have_spllited_data:
       return image_dict, train_roots_dict, val_roots_dict, test_roots_dict
    else:
       return image_dict



if Tube == "":
    check_tube = True
else:
    check_tube = False


if txt_path == "":
    points_files_dir = os.listdir(all_txt_paths)
    for x in points_files_dir:
        txt_path = os.path.join(all_txt_paths, x)
        image_dict, train_image_dict, val_image_dict, test_image_dict = parse_txt(image_dict, train_image_dict, val_image_dict, test_image_dict, txt_path, check_tube, Tube)
else:
    if have_spllited_data:
        image_dict, train_image_dict, val_image_dict, test_image_dict = parse_txt(image_dict, txt_path, check_tube, Tube,
                                                                                  train_image_dict, val_image_dict, test_image_dict)
    else:
        image_dict = parse_txt(image_dict, txt_path, check_tube, Tube)


########################################################################################################

# add avg and std of diameter per image

for key in image_dict.keys():
    # print(key)
    # if key=="3_31_6":
    #     a=1
    if len(image_dict[key]["roots_dia_values"])>0:
        image_dict[key]["roots_dia_mean"] = np.mean(image_dict[key]["roots_dia_values"])
    if image_dict[key]["roots_num"]>1:
        image_dict[key]["roots_dia_std"] = np.std(image_dict[key]["roots_dia_values"])

if have_spllited_data:

    for key in train_roots_dict.keys():
        if len(train_roots_dict[key]["roots_dia_values"]) > 0:
            train_roots_dict[key]["roots_dia_mean"] = np.mean(train_roots_dict[key]["roots_dia_values"])
        if train_roots_dict[key]["roots_num"]>1:
            train_roots_dict[key]["roots_dia_std"] = np.std(train_roots_dict[key]["roots_dia_values"])


    for key in val_roots_dict.keys():
        if len(val_roots_dict[key]["roots_dia_values"]) > 0:
            val_roots_dict[key]["roots_dia_mean"] = np.mean(val_roots_dict[key]["roots_dia_values"])
        if val_roots_dict[key]["roots_num"]>1:
            val_roots_dict[key]["roots_dia_std"] = np.std(val_roots_dict[key]["roots_dia_values"])


    for key in test_roots_dict.keys():
        if len(test_roots_dict[key]["roots_dia_values"]) > 0:
            test_roots_dict[key]["roots_dia_mean"] = np.mean(test_roots_dict[key]["roots_dia_values"])
        if test_roots_dict[key]["roots_num"] > 1:
            test_roots_dict[key]["roots_dia_std"] = np.std(test_roots_dict[key]["roots_dia_values"])


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
    #a file with all the data, not splitted
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
