import os
import shutil
import pandas as pd
import csv


data_dir = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names", "Tube_7_prev")

dir_list = os.listdir(data_dir)
for x in dir_list:
    # read images names
    if x.endswith(".jpg"):
        if "L_" in x:
            os.remove(os.path.join(data_dir, x))