import os
import sys
import csv
import cv2
import torch

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

import config

########################################################################################################################

# Check if GPU is available
config.General.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set GPU if available
if config.General.device.type == 'cuda':
    current_gpu = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))
else:
    print('Running on cpu')

########################################################################################################################
#dataset_name = "Train" #"Tube 43" #"MELON 2018_tube 17"  # "TOMATO 2019_tube 4" #"MELON 2019_tube 9" #"PEPPER 2021_tube 8" # "MELON 2018_tube 17" #"CORN 2020_tube 16"  #"Parthasarathi_tomato_Tube 9"  #"Rootfly_cam3"  #"new_auto_cam"  #"Pepper 2021" #"Corn 2020" #"Tomato 2020" #"Tomato 2019" #"Pepper 2021"  "Melon 2019", "Melon 2018"

#type = "GFP" #"RGB" #"GFP"
#time="168h"
#dir = "Train"
data_path = os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily", "annotations",
                         "splitted", "sub_Test")
                         #"daily_by_cam", "cam9\\Tube_9_wrong sessions", "additional_annotations") #"annotations\\splitted","sub_Val")
                         #"processed_names", "Tube_7", "additional_annotations_correct names", "cam13")#"Field2\\cam14\\Tube_2", #   "additional_annotations_correct names\\cam9")


    #os.path.join("D:\\Faina", "Roots", "Sharon", "Hatzeva_all_images_daily\\processed_names",
     #                    "Field2\\cam13\\Tube_1","additional_annotations", "sess29"))#"annotations\\all_additional_anns")
#("D:\\Faina\\Roots","Xu\\with_anns\\renamed_random_jpg_b","all_images_with_anns", type, "splitted", "sub_"+dir)
#("D:\\Faina\\Roots","Xu\\with_anns\\renamed_random_jpg_b",time , type+" roots")#"GFP roots") # "Sharon", "Hatzeva 2024-2025-Faina"
#os.path.join("E:\\roots_project", "Grapes_K_fold", "K_1", "sub_"+dataset_name+"_1")#"Root hairs","GFP DROUGHT\\correctionsgfpdrought\\crops")
                         #"all\\sub_"+dataset_name)#"GFP-control", "crops") #"Root hairs", "GFP DROUGHT" "GFP-control"
#os.path.join("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Tube 22_renamed")
    #("D:/Faina/roots_project", "Dataset for root color model","17_1_data\\For root color model\\all images")
                         #"Selected images from Brian_Grapevine\\Rootfly_subfolders_Grapevine roots"
                         #os.path.join("D:/Faina/roots_project", "Dataset for root color model\\Selected images from Brian_Grapevine",
                         #"Rootfly_subfolders_Grapevine roots")
                        #os.path.join("D:/Faina/roots_project", "Autocam image for CNN model", "Images for training\\Training dataset_automated camera\\original\\sub_Train")
                        #"D:\\Faina\\roots_project\\manual_camera\\Sep 22\\NEW- manual camera\\MELON 2018_tube 17"
                        #os.path.join("D:/Faina/roots_project", "Autocam image for CNN model", "Images for training\\Training dataset_automated camera\\original\\sub_Train")
                         #"Chosen images for root diameter", dataset_name) #"Aug 22")
                         #"manual_camera", "July_22" , dataset_name ) #, "sub_Train")
                         #"Autocam image for CNN model", "Images for training\\Training dataset_automated camera\\original")
                        #Training dataset_Rootfly")
                        #os.path.join("D:\\Faina\\roots_project", dataset_name, "June 22") #"manual_camera", "13_5_22\\Correction after visual inspection_20220513", dataset_name) # "corrected"

points_data_path = os.path.join(data_path, "Test_pointsOutput.csv") #"combined_pointsOutput.csv") #"combined_pointsOutput.csv") # dir+"_pointsOutput.csv" #dataset_name+"_pointsOutput_limit_5.csv") #"pointsOutput_hairs_single_3.csv") #"pointsOutput_main.csv" #"pointsOutput_hairs.csv"#"Train_pointsOutput.csv") #dataset_name+"_pointsOutput.csv") #"sub_Test", "Test_pointsOutput.csv") #"pointsOutput_June.csv") #"Test_pointsOutput_June.csv"

img_path = data_path #os.path.join(data_path, "raw_images") #os.path.join(data_path, "sub_Test")  #os.path.join("D:\\Faina\\roots_project", dataset_name, "processed_img")
input_TRL = os.path.join(data_path, "Test.csv")#"combined_TRL.csv") #"combined_TRL.csv")
# dir+".csv" #dataset_name+"_limit_5.csv") #"count_hairs_3.csv")
                         #dataset_name+"_count_hairsPerTRL.csv") #"TRL_main.csv") #"TRL_hairs.csv" #"TRL.csv") #"autoCam3_test_TRL_raw.csv") #"Train.csv") #dataset_name+"_TRL.csv") #"sub_Test", "Test.csv") #dataset_name+"_TRL.csv")

########################################################################################################################

move_points = False

rezise_points_to_480_640 = False

save_cropped = False

put_text = True
########################################################################################################################


output_images_path = os.path.join(data_path, "draw_GT") #"draw_roots_hairs" #"draw_main_roots"#"draw_new_GT" #"draw_autoCam_Val") #"draw_roots_hairs" "draw_main_roots"
os.makedirs(output_images_path, exist_ok=True)

withRoot_images_path = os.path.join(output_images_path, "with_Root") #"with_Root" #"with_Root_hairs"
os.makedirs(withRoot_images_path, exist_ok=True)

noRoot_images_path = os.path.join(output_images_path, "no_Root") #"no_Root" #"no_Root_hairs"
os.makedirs(noRoot_images_path, exist_ok=True)

########################################################################################################################
images_match_file = "" #os.path.join(data_path, "images_match_June.csv")
draw_orig_on_procc = False
processed_to_orig = {}
if draw_orig_on_procc:
    with open(images_match_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            p = rows[0]
            o = rows[1].split("\\")[1]

            processed_to_orig[p] = o

########################################################################################################################

# get TRL values
img_dict = {}

with open(input_TRL) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_dict[row[0]] = round(float(row[1]),2)


#img032_2021-08-16_02-54-06.jpg
points_dict = {}
with open(points_data_path, mode='r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        #print(row[0])
        points_dict[row[0]] =[]
        points_num = int((len(row)-1)/2)
        if len(row)>1:
            i=1
            count = 0
            while count < points_num and (row[i]!=""):
                points_dict[row[0]].append((int(row[i]), int(row[i+1])))
                i+=2
                count += 1


dir_list = os.listdir(img_path)

for x in dir_list:
    # read images names
    if x.endswith(".jpg") or x.endswith(".png"):
        print("Draw points on:", x)

        image = cv2.imread(os.path.join(img_path, x))
        if save_cropped:
            image = image[21:471, 16:, :]

        if draw_orig_on_procc:
            name= processed_to_orig[x]
        else:
            name=x

        if name in points_dict.keys():
            if len(points_dict[name]) > 0:
                points = points_dict[name]

                if rezise_points_to_480_640:
                    H_ratio = 480/1944 #image.shape[0] / 480
                    W_ratio = 640/2592 #image.shape[1] / 640

                for p in points:
                    if move_points:
                        p = (p[0]-16, p[1]-21)


                    if rezise_points_to_480_640:
                        image = cv2.circle(image, (int(p[0]*W_ratio), int(p[1]*H_ratio)), radius = 2, color = (255, 0, 0), thickness = -1)
                    else:
                        image = cv2.circle(image, p, radius=7, color=(255, 0, 0), thickness=-1) #radius=7 radius=4 #(255, 0, 0) radius=2, color=(0, 255, 0),


            # Displaying the image
            #cv2.imshow(image)
            #cv2.putText(image, str(img_dict[x]), org=(100,100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,0,0), thickness =1) #"TRL = "+

            if put_text:
                cv2.putText(image, "GT TRL = "+str(img_dict[x]), org=(100, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                            color=(255, 255, 255), thickness=3)  # "TRL = "+

            if os.path.exists(os.path.join(output_images_path, "dont use")):
                if os.path.exists(os.path.join(output_images_path, "no points")):
                    if os.path.exists(os.path.join(output_images_path, "no points", "points_"+x)):
                        new_path = os.path.join(output_images_path, "new no points")
                        os.makedirs(new_path, exist_ok=True)
                        cv2.imwrite(os.path.join(new_path, "points_" + x), image)

                    elif not os.path.exists(os.path.join(output_images_path, "dont use", "points_"+x)):
                        if img_dict[x] > 0:
                            cv2.imwrite(os.path.join(withRoot_images_path, "points_" + x), image)
                        else:
                            cv2.imwrite(os.path.join(noRoot_images_path, "points_" + x), image)



            else:
                if img_dict[x] > 0:
                    cv2.imwrite(os.path.join(withRoot_images_path, "points_" + x), image)
                else:
                    cv2.imwrite(os.path.join(noRoot_images_path, "points_" + x), image)
















