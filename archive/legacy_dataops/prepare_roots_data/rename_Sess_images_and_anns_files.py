import os
import shutil
import pandas as pd
import csv



data_dir = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names", "Tube_7") #\\Field2\\cam14\\Tube_2")

prev_anns_dir = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names", "cam13\\Tube_7", "additional_annotations") # "sess7")
corrected_anns_dir = os.path.join(data_dir, "additional_annotations_correct names", "cam13") #"sess7")
os.makedirs(corrected_anns_dir, exist_ok=True)

names_file = os.path.join(data_dir,"PrevToNewName.xlsx")


# Read the Excel file
df = pd.read_excel(names_file)

prev_TRL_file = os.path.join(prev_anns_dir, "cam13_Tube_7_TRL.csv")#"Field2_cam14_Tube_1_TRL.csv")
new_TRL_file = os.path.join(corrected_anns_dir, "cam13_Tube_7_TRL.csv") #"Field2_cam14_Tube_2_TRL.csv")

prev_points_file = os.path.join(prev_anns_dir, "cam13_Tube_7_pointsOutput.csv")#"Field2_cam14_Tube_1_pointsOutput.csv")
new_points_file = os.path.join(corrected_anns_dir, "cam13_Tube_7_pointsOutput.csv") #"Field2_cam14_Tube_2_pointsOutput.csv")


# create correct TRL file and copy images with the correct name to the corrected anns dir
with open(prev_TRL_file, 'r', newline='', encoding='utf-8') as infile, \
    open(new_TRL_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # # Make sure the row has enough columns
        # while len(row) <= 1:
        #     row.append('')

        prev_name = row[0]
        new_name = df.loc[df['prev_name'] == prev_name, 'new_name'].values[0]

        if "L_" in new_name:
            new_name_split = new_name.split("L_")
            new_name_2 = new_name_split[0]+"L"+new_name_split[1]
            if os.path.exists(os.path.join(data_dir, new_name)):
                shutil.move(os.path.join(data_dir, new_name), os.path.join(corrected_anns_dir, new_name_2))
            row[0] = new_name_2

        else:
            if os.path.exists(os.path.join(data_dir, new_name)):
                shutil.move(os.path.join(data_dir, new_name), os.path.join(corrected_anns_dir, new_name))
            row[0] = new_name
        writer.writerow(row)

# create correct points file
with open(prev_points_file, 'r', newline='', encoding='utf-8') as infile, \
        open(new_points_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        prev_name = row[0]
        new_name = df.loc[df['prev_name'] == prev_name, 'new_name'].values[0]

        if "L_" in new_name:
            new_name_split = new_name.split("L_")
            new_name = new_name_split[0] + "L" + new_name_split[1]

        row[0] = new_name
        writer.writerow(row)