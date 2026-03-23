import os
import csv
import shutil
import json
import pandas as pd
from pathlib import Path





data_path = os.path.join("D:\\Faina\\Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names\\Tube_11", "Results\\Vis") #"bi-weekly","Tube 11_biweekly", "Results\\Vis")
#file_path = os.path.join(data_path, "Tube_1_Cam13_lastDate_2025-02-03_sess_56_images_match.csv") #"Tube_1_Cam14_lastDate_2024-10-12_sess_9_images_match.csv") #"Tube_11_Cam9_lastDate_2025-04-22_sess_135_images_match.csv" "Tube_9_Cam9_lastDate_2024-12-08_sess_14_images_match.csv" # "Tube_7_Cam13_lastDate_2025-04-27_sess_56_images_match.csv" "Tube_2_Cam2_lastDate_2025-04-27_sess_196_images_match.csv")
target_dir = os.path.join("D:\\Faina\\Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names\\Tube_11", "Results\\Vis_DAP 90") #"bi-weekly", "Tube 11_biweekly", "Results\\Vis_DAP 90") #data_path #os.path.join(data_path, "RGB_Joined datasets") # "Joined datasets" #"Three_datasets_detection")


#df = pd.read_csv(file_path)
#
# for index, row in df.iterrows():
#     New_name = row['New_name']
#
#     if not os.path.exists(os.path.join(data_path,New_name)):
#         Orig_name = row['Orig_name']
#         orig_image_path = row['Orig_im_path']
#         path = Path(os.path.join("G:\\Hatzeva 2024-2025\\Rootcam",orig_image_path))
#
#         print(Orig_name, New_name)
#
#         shutil.copyfile(os.path.join(path, Orig_name), os.path.join(data_path, New_name))




dir_list = os.listdir(data_path)
# copy images
for x in dir_list:
    # read images names
    if x.endswith(".png"):#(".jpg"):

        if "001_map" in x:
            shutil.copy(os.path.join(data_path,x), os.path.join(target_dir,x))

