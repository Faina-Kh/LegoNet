import csv
import os
import pandas as pd
import json

#######################################################################################################################

# Define paths

dataset_name = "Tube 43"
#"MELON 2018_tube 40" #"MELON 2018_tube 17" #"PEPPER 2021_tube 8" #"tomato 2020_tube 13" #"TOMATO 2019_tube 4" #"MELON 2019_tube 9" #"MELON 2019_T9" #"MELON 2018_tube 17" #"CORN 2020_tube 16"  #"Parthasarathi_tomato_Tube 9"  #"Rootfly_cam3\\Aug 22"
#"Tomato 2020"  #"new_auto_cam" #"Tomato 2019" #"Melon 2018" # "Tomato 2020" # "Pepper 2021" # "Melon 2019" #"Corn 2020"
#"Rootfly_cam3"
data_path = os.path.join("D:/Faina/roots_project", "Chosen images for root diameter", dataset_name)
                         #"manual_camera" , "Sep 22", "NEW- manual camera", dataset_name)
#os.path.join("D:/Faina/roots_project", "manual_camera", "July_22" , dataset_name)
#os.path.join("D:\Faina\\roots_project", "Autocam image for CNN model", "Images for training\\Training dataset_automated camera")
#os.path.join("D:/Faina/roots_project", "manual_camera\\13_5_22\\Correction after visual inspection_20220513" , dataset_name)
# generate paths to raw files

path_to_files = []

raw_files = ["Session 2 tube 43.csv", "Session 5 tube 43.csv", "Session 9 tube 43.csv"]
#["Melon 2018_Tube 40.csv"] #["Melon 2018_Tube 17_20221205.csv"] #["Melon 2018_Tube 17.csv"]  #["Tomato 2020_Tube 13.csv"]  #["Tomato 2019_Tube 4.csv"] #["Pepper 2021_Tube 8.csv"] #["Melon 2019_Tube 9.csv"] #["Corn 2020_Tube 16.csv"]
#["Kaining_Pepper_2021.csv"] #["tomato 2020.csv"] #["T1-28_Melon_2018 -T17.csv"]  #["T1-28_T57-58_melon_2019.csv"] #["Kaining_Pepper_2021.csv"] #["T1-24_tomato_2019.csv"] #["Corn-2020.csv"]

for file in raw_files:
       path_to_files.append(os.path.join(data_path, file)) #"June 22",

#path_to_files.append(os.path.join(data_path, "cam3_every 5 days_for model testing_corrected on 20220811.csv"))
#"cam3_every 5 days_for model testing_corrected on 20220810.csv"
#os.path.join(data_path, dataset_name+".csv"))

processed_img_path = os.path.join(data_path) #, "chosen_images") #"images")
    #os.path.join("D:/Faina/roots_project", "Rootfly_cam3\\processed_img") #data_path
    #os.path.join(data_path, "Images for training model_resized")
    #os.path.join("D:/Faina/roots_project",  "manual_camera", "13_5_22\\Correction after visual inspection_20220513", "images", dataset_name) #os.path.join(data_path, "processed_img")


# output files
output_csv_file = os.path.join(data_path, "TRL.csv") #"autoCam3_test"+ "_TRL.csv")  #os.path.join(data_path, "June 22", "TRLJune.csv")
json_output_path = os.path.join(data_path, "roots_info.txt") #"autoCam3_test"+ "_roots_info.txt")  #"June 22", )

with_time_format = True # True or False

generate_csv = True
generate_json = True


####################################################################################################################


file_dict = {}

root_dict ={}

dia_avg_per_image = {}

# parse raw file data
for path in path_to_files:
    data = pd.read_csv(path)
    header = data.columns

    for i in range(len(data)):
        T = str(int(data['Tube'][i]))
        L = str(int(data['Window'][i]))

        RootID =str(data['RootID'][i])

        for j in range(len(header)):
            H = header[j]
            if "Length date" in H:
                #print(H)
                S = H.split("(")[1].split(")")[0]
                if data[H][i]!=" ":
                    if float(data[H][i]) > 0:
                        length = float(data[H][i])
                        current_name = T+'_'+L+'_'+S    #current_key = str(current_T)+ "_" + str(current_L)+ "_" + str(current_Sess)

                        if current_name == "16_19_4":
                            a=1

                        if current_name in file_dict.keys():
                            file_dict[current_name] += length
                        else:
                            file_dict[current_name] = length


                        if current_name not in root_dict.keys():
                            root_dict[current_name]={}
                            root_dict[current_name]["RootID"+"_"+RootID] = {}

                        if "RootID"+"_"+RootID not in root_dict[current_name].keys():
                            root_dict[current_name]["RootID" + "_" + RootID] = {}

                        root_dict[current_name]["RootID" + "_" + RootID]["Length"] = length


            if "Diameter date" in H:
                S = H.split("(")[1].split(")")[0]
                if data[H][i] != " ":
                    if float(data[H][i]) > 0:
                        Diameter = float(data[H][i])
                        current_name = T + '_' + L + '_' + S
                        #print(current_name)
                        if current_name not in root_dict.keys():
                            root_dict[current_name]={}
                            root_dict[current_name]["RootID" + "_" + RootID] = {}

                        if "RootID"+"_"+RootID not in root_dict[current_name].keys():
                            root_dict[current_name]["RootID" + "_" + RootID] = {}

                        root_dict[current_name]["RootID"+"_"+RootID]["Diameter"] = Diameter

                        if current_name not in dia_avg_per_image.keys():
                            dia_avg_per_image[current_name]={}


csv_rows = []
root_dict_filtered = {}

# match each image to the TRL found in the raw file
def get_data(processed_img_path, csv_rows, root_dict_filtered):
    dir_list = os.listdir(processed_img_path)
    for x in dir_list:
        # read images names
        if x.endswith(".jpg"):
            # Prints only text file present in My Folder
            print(x)

            # if "(1)" in x:
            #     os.remove(os.path.join(processed_img_path, x))
            #     continue

            # create the row for the output file
            myrow = []
            myrow.append(x)

            name_split = x.split("_")
            current_T = name_split[1]
            current_L = name_split[2]

            if with_time_format:
                current_Sess = name_split[5]
            else:
                current_Sess = name_split[4].split('.jpg')[0]

            Tube_int = int(current_T.split('T')[1])
            Loc_int = int(current_L.split('L')[1])
            Sess_int = int(current_Sess)

            # if 'T1-T24 -raw data' in raw_file:
            #      if Tube_int>24:
            #          continue

            current_key = str(Tube_int) + "_" + str(Loc_int) + "_" + str(Sess_int)

            #if current_key=="16_19_4":
            #    a=1


            print(current_key)

            if current_key in file_dict.keys():
                current_value = file_dict[current_key]
                root_dict_filtered[current_key] = root_dict[current_key]

            else:
                current_value = 0.0
                root_dict_filtered[current_key] = []

            myrow.append(current_value)
            csv_rows.append(myrow)
    return csv_rows, root_dict_filtered


if dataset_name == "new_auto_cam":
    dir_list = os.listdir(processed_img_path)
    for dir in dir_list:
        if dir != '160':
            dir_path = os.path.join(processed_img_path, dir)
            csv_rows, root_dict_filtered = get_data(dir_path, csv_rows, root_dict_filtered)

else:
    csv_rows, root_dict_filtered = get_data(processed_img_path, csv_rows, root_dict_filtered)



if generate_csv:
    f = open(output_csv_file, 'w', newline='')
    with f:
        writer = csv.writer(f)
        for myrow in csv_rows:
            writer.writerow(myrow)



if generate_json:
    # print roots data to json
    with open(json_output_path, "w") as outfile:
        json.dump(root_dict_filtered, outfile)

print("Done")
