import os
import shutil
import csv


data_path = "D:\\Faina\\roots_project\\Datasets\\Dataset 2\\sub_Train"
input_TRL = os.path.join(data_path, "Train_TRL.csv")

images= []
with open(input_TRL) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        images.append(row[0])



dir_list = os.listdir(data_path)
# copy images
for x in dir_list:
    # read images names
    if x.endswith(".jpg"):
        if x not in images:
            os.remove(os.path.join(data_path, x))



