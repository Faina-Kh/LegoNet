import os
import shutil
import pandas as pd
import csv



data_dir = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names", "Tube_7") #\\Field2\\cam14\\Tube_2")
new_data_dir = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names", "Tube_7")
os.makedirs(new_data_dir, exist_ok=True)

correct_anns_dir = False
copy_images_from_data_dir = True

if correct_anns_dir:
    prev_anns_dir = os.path.join(data_dir, "additional_annotations", "sess7")
    new_anns_dir = os.path.join(new_data_dir, "additional_annotations", "sess7")
    os.makedirs(new_anns_dir, exist_ok=True)

    prev_TRL_file = os.path.join(prev_anns_dir, "Field2_cam13_Tube_1_TRL.csv")#"Field2_cam14_Tube_1_TRL.csv")
    new_TRL_file = os.path.join(new_anns_dir, "Field2_cam13_Tube_1_TRL.csv") #"Field2_cam14_Tube_2_TRL.csv")

    prev_points_file = os.path.join(prev_anns_dir, "Field2_cam13_Tube_1_pointsOutput.csv")#"Field2_cam14_Tube_1_pointsOutput.csv")
    new_points_file = os.path.join(new_anns_dir, "Field2_cam13_Tube_1_pointsOutput.csv") #"Field2_cam14_Tube_2_pointsOutput.csv")

# copy all images to the new folder without the "L_"
if copy_images_from_data_dir:
    dir_list = os.listdir(data_dir)
    for x in dir_list:
        # read images names
        if x.endswith(".jpg"):
            if "L_" in x:
                splitted = x.split("L_")
                new_name = splitted[0]+"L"+splitted[1]
                shutil.copy(os.path.join(data_dir, x), os.path.join(new_data_dir, new_name))


if correct_anns_dir:
    # handle annotated data
    # create correct TRL file and copy images with the correct name to the corrected anns dir
    with open(prev_TRL_file, 'r', newline='', encoding='utf-8') as infile, \
        open(new_TRL_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            prev_name = row[0]

            if "L_" in prev_name:
                new_name_split = prev_name.split("L_")
                new_name = new_name_split[0]+"L"+new_name_split[1]

                if os.path.exists(os.path.join(new_data_dir, new_name)):
                    shutil.move(os.path.join(new_data_dir, new_name), os.path.join(new_anns_dir, new_name))
                row[0] = new_name

            else:
                if os.path.exists(os.path.join(new_data_dir, new_name)):
                    shutil.move(os.path.join(new_data_dir, new_name), os.path.join(new_anns_dir, new_name))
                row[0] = new_name
            writer.writerow(row)

    # create correct points file
    with open(prev_points_file, 'r', newline='', encoding='utf-8') as infile, \
            open(new_points_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            prev_name = row[0]

            if "L_" in prev_name:
                new_name_split = prev_name.split("L_")
                new_name = new_name_split[0] + "L" + new_name_split[1]
            else:
                new_name = prev_name


            row[0] = new_name
            writer.writerow(row)