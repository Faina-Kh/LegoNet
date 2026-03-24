import os
import cv2

#################################################################################################################

current_gpu = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# paths to existing data
dataset_name = "" #"Tube 43" #"MELON 2018_tube 40"  #"PEPPER 2021_tube 8"  #"tomato 2020_tube 13" # #"TOMATO 2019_tube 4" #"MELON 2019_tube 9" #"PEPPER 2021_tube 8" #"MELON 2018_tube 17" #"CORN 2020_tube 16"
data_path = os.path.join("D:\\Faina\\roots_project", "SHARONARAVA2022_images")
                         #"Chosen images for root diameter", dataset_name)#"manual_camera", "Sep 22", "NEW- manual camera", dataset_name)

#img_path = os.path.join(data_path, "chosen_images") #"splitted_data\\processed_withRoot")

save_path = os.path.join(data_path, "without frame images") #os.path.join(data_path, "pepperm2021-splitted_data", "all_Cropped") #"split images-roots", "splitted_data- tomato 2020", 'resized')  #cropped
os.makedirs(save_path, exist_ok=True)

to_crop = True
to_resize = False


dir_list = os.listdir(data_path)

for x in dir_list:
    # read images names
    if x.endswith(".jpg"):
        print(x)
        image = cv2.imread(os.path.join(data_path, x))

        if to_crop:
            image = image[21:471, 16:,:]

        if to_resize:
            dim = (640, 480)   #(2592,1944)
            image = cv2.resize(image, dim)

        cv2.imwrite(os.path.join(save_path, x), image)








