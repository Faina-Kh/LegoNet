import csv
import os
import pandas as pd


##################################################################################

# Define paths

dataset_name = "" #"5_5_22\\Corrected annotation\\pepper_2021" #"corrected\\Tomato 2020"
data_path = os.path.join("D:/Faina/roots_project",  "SELHAUSEN_farming", "TEST", "01_RAW_DATA\\RUT\\2017\\20170713\\RUT20170713_original")#dataset_name)

raw_files = ["kaining_pepper_2021.csv"]

output_csv_file = os.path.join(data_path, "TRL.csv")

with_time_format = True # True or False

##################################################################################

# generate paths to raw files
path_to_files = []
for file in raw_files:
    path_to_files.append(os.path.join(data_path, file))

file_dict = {}

# parse raw file data
for path in path_to_files:
    data = pd.read_csv(path)
    header = data.columns

    for i in range(len(data)):
        T = str(data['Tube'][i])
        L = str(data['Window'][i])
        for j in range(len(header)):
            H = header[j]
            if "Length date" in H:
                #print(H)
                S = H.split("(")[1].split(")")[0]
                if data[H][i]!=" ":
                    if float(data[H][i]) > 0:
                        length = float(data[H][i])
                        current_name = T+'_'+L+'_'+S
                        if current_name in file_dict.keys():
                            file_dict[current_name] += length
                        else:
                            file_dict[current_name] = length


# match each image to the TRL found in the raw file
dir_list = os.listdir(data_path)

f = open(output_csv_file, 'w', newline='')
with f:
    writer = csv.writer(f)
    for x in dir_list:
        # read images names
        if x.endswith(".jpg"):
            # Prints only text file present in My Folder
            print(x)

            #create the row for the output file
            myrow = []
            myrow.append(x)

            name_split = x.split("_")
            current_T = name_split[1]
            current_L = name_split[2]

            if with_time_format:
                current_Sess = name_split[5]
            else:
                current_Sess = name_split[4].split('.jpg')[0]

            current_name = current_T + "_" + current_L + "_" + current_Sess

            Tube_int = int(current_T.split('T')[1])
            Loc_int = int(current_L.split('L')[1])
            Sess_int = int(current_Sess)

            # if 'T1-T24 -raw data' in raw_file:
            #      if Tube_int>24:
            #          continue

            current_key = str(Tube_int)+ "_" + str(Loc_int)+ "_" +str(Sess_int)
            if current_key in file_dict.keys():
                current_value = file_dict[current_key]
            else:
                current_value = 0.0

            myrow.append(current_value)
            writer.writerow(myrow)

print("Done")
