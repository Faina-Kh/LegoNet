import os
import csv
import cv2

name_match_file = os.path.join("D:\\Faina\\roots_project", "Sharon auto\\raw_images\\cam2-T002\\Cam2data", "images_match_2.csv")
                  #\\Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\Autocam3_files\\images_match_newAutoCam.csv"
#"D:\\Faina\\roots_project\\Rootfly_cam3\\Aug 22\\name_match.csv"

input_file = os.path.join("D:\\Faina\\roots_project","Sharon auto\\raw_images\\cam2-T002\\Cam2data", "processed_TRL.csv")
                          #"Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\, "original","sub_Train", "Train.csv")
                           #"Rootfly_cam3", "Aug 22", "Results" , "Diameter_10Loss", "test diameter.csv")
                           #"TRL only", "test results of 11_8_22.csv")
                          #"D:\\Faina\\roots_project\\Rootfly_cam3\\Aug 22", "Results\\TRL only", "test results 2.csv"
                          # "autoCam3_test_TRL.csv"

output_file = os.path.join("D:\\Faina\\roots_project","Sharon auto\\raw_images\\cam2-T002\\Cam2data","TRL_raw.csv")
                           #"Autocam image for CNN model\\Images for training\\Training dataset_automated camera\\","original","sub_Train", "Train_TRL_with_names.csv")
                           #"Rootfly_cam3\\Aug 22","Results", "Diameter_10Loss", "diameter_results_with_names.csv") #"autoCam3_test_TRL_raw.csv" # "Results\\TRL only", "results_with_names.csv"

have_title = False
creat_raw_TRL = False
create_results_match_file = True

########################################################################################################################
names_dict = {}
with open(name_match_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if creat_raw_TRL:
            names_dict[row[0]] = row[1]
        elif create_results_match_file:
            raw_name = row[1].split("\\")[1]
            #names_dict[row[1]] = row[0]
            raw_name = row[1].split("\\")[1]
            names_dict[raw_name] = row[0]


new_rows = []
count=0
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        new_row = []
        if create_results_match_file:
            if count ==0:
                if have_title:
                    new_row.append("rootfly_name")
                    # copy titles
                    for i in range(len(row)):
                        new_row.append(row[i])
                    new_rows.append(new_row)
                count = 1
            else:
                print(row[0].split(" ")[0]+".jpg")
                rootfly_name = names_dict[row[0].split(" ")[0]]#+".jpg"]
                new_row.append(rootfly_name)
                for i in range(len(row)):
                    new_row.append(row[i])
                new_rows.append(new_row)

        elif creat_raw_TRL:
            raw_name = names_dict[row[0]]
            new_row.append(raw_name)
            TRL = row[1]
            new_row.append(TRL)
            new_rows.append(new_row)


f = open(output_file, 'w', newline='')
with f:
    writer = csv.writer(f)
    for row in new_rows:
        writer.writerow(row)
