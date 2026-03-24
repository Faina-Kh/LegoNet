import os
import shutil


data_path = os.path.join("", "Tube 40" )

sessions= [2, 5, 9]


copy_to = os.path.join(data_path, "chosen_images")
os.makedirs(copy_to, exist_ok=True)


dir_list = os.listdir(data_path)
# copy images
for x in dir_list:
    # read images names
    if x.endswith(".jpg"):

        name_split = x.split("_")
        #current_L = name_split[2]
        current_Sess = int(name_split[5])

        if current_Sess in sessions:
            shutil.copyfile(os.path.join(data_path, x), os.path.join(copy_to, x))



