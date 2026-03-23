import numpy as np
import csv
import random
from random import sample
import os
##############################################################################################################################################

random.seed(0)
np.random.seed(0)

#################################################################################################################

current_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################

##############################################################################################################################################
only_with_roots = True

image_num = 100

with_roots_num = "" #10 #70
no_roots_num = "" #7

input_TRL =  os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "sub_Train\\Train.csv")
#"July_22\\all_manual_2", "sub_Test\\Test.csv") #"D:\\Faina\\roots_project", "manual_camera\\July_22\\all_manual_2", "sub_Train\\Train.csv"
input_points = os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "sub_Train\\Train_pointsOutput.csv")
                            #"July_22\\all_manual_2", "sub_Test\\Test_pointsOutput.csv")#"D:\\Faina\\roots_project\\manual_camera\\July_22\\all_manual_2\\sub_Train\\Train_pointsOutput.csv"

output_TRL =  os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "sub_Train", "Train_additional_"+str(image_num)+".csv")
                           #"July_22\\all_manual_2", "sub_Test\\Test_compareToAuto.csv") #+str(image_num)+"_with roots.csv" #"D:\\Faina\\roots_project\\manual_camera\\July_22\\all_manual_2\\sub_Train\\Train_additional_20 with roots.csv"
output_points = os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project",  "Grapevine_data_all", "sub_Train", "Train_pointsOutput_additional_"+str(image_num)+".csv")
                             #"July_22\\all_manual_2", "sub_Test\\Test_pointsOutput_compareToAuto.csv")  #_additional_"+str(image_num)+"_with roots.csv") #"D:\\Faina\\roots_project\\manual_camera\\July_22\\all_manual_2\\sub_Train\\Train_pointsOutput_additional_20 with roots.csv"

################################################################################################################################################

image_names = []
TRL_dict = {}
points_dict={}

if not only_with_roots:
    image_names_no_roots = []
    TRL_dict_no_roots = {}
    points_dict_no_roots = {}

#random sets with and without roots

with open(input_TRL) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if float(row[1])>0:
            image_names.append(row[0])
            TRL_dict[row[0]] = row

        elif not only_with_roots:
            image_names_no_roots.append(row[0])
            TRL_dict_no_roots[row[0]] = row

with open(input_points) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if len(row)>1:
            points_dict[row[0]] = row
        elif not only_with_roots:
            points_dict_no_roots[row[0]] = row



if only_with_roots:
    chosen_names = sample(image_names, image_num)
else:
    chosen_names = sample(image_names, with_roots_num)
    chosen_names_no_roots = sample(image_names_no_roots, no_roots_num)



output1 = open(output_TRL, 'w', newline='')
with output1:
    writer = csv.writer(output1)
    for name in chosen_names:
        writer.writerow(TRL_dict[name])

    if not only_with_roots:
        for name in chosen_names_no_roots:
            writer.writerow(TRL_dict_no_roots[name])


output2 = open(output_points, 'w', newline='')
with output2:
    writer = csv.writer(output2)
    for name in chosen_names:
        writer.writerow(points_dict[name])

    if not only_with_roots:
        for name in chosen_names_no_roots:
            writer.writerow(points_dict_no_roots[name])

    if not only_with_roots:
        for name in chosen_names_no_roots:
            writer.writerow(points_dict_no_roots[name])
