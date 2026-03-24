import os
import csv
import shutil
import json


data_path = os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Tubes 21-22") #"Tubes 19-20")

target_dir = os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Tube 22_renamed")
os.makedirs(target_dir, exist_ok=True)

###############################################################################################################################

dir_list = os.listdir(data_path)

# copy images
for x in dir_list:
    # read images names
    if x.endswith(".jpg"):
        if "T022" in x:
            image_name = x.split("_")
            new_name = image_name[1]+"_"+image_name[2]+"_"+image_name[3]+"_"+image_name[4]+"_"+image_name[5]+".jpg"
            shutil.copyfile(os.path.join(data_path, x), os.path.join(target_dir, new_name))