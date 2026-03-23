# Faina Khoroshevsky, 304673056


import os
import shutil

import numpy as np


def main(dataset_name):

    data_dir = "../data/"+ dataset_name

    my_labels_dir = data_dir + '/labelsTr'
    my_images_dir = data_dir + '/imagesTr'

    sub_img_Train_Val = os.path.join(data_dir, "img_sub_Train_Val")
    sub_img_Test = os.path.join(data_dir, "img_sub_Test")

    sub_label_Train_Val = os.path.join(data_dir, "label_sub_Train_Val")
    sub_label_Test = os.path.join(data_dir, "label_sub_Test")

    # create folders if don't exist
    os.makedirs(sub_img_Train_Val, exist_ok=True)
    os.makedirs(sub_img_Test, exist_ok=True)
    os.makedirs(sub_label_Train_Val, exist_ok=True)
    os.makedirs(sub_label_Test, exist_ok=True)

    if dataset_name == "Task01_BrainTumour":  #"Task01_BrainTumour" , "Task04_Hippocampus"
        numTraining = 484
    elif dataset_name == "Task04_Hippocampus":
        numTraining = 260

    np.random.seed(0)
    count=0
    for (dirpath, dirnames, filenames) in os.walk(my_images_dir):
        for filename in filenames:
            if filename.split("_")[0] == ".":
                continue

            filepath = os.path.join(dirpath, filename)
            labelpath = os.path.join(my_labels_dir, filename)

            count+=1

            # 20% of data will go to testing
            rnd = np.random.random()

            # copy images
            print("copy file", count, "out of", numTraining, ":", filename)
            if rnd <=0.2:
                dst_img_file = os.path.join(sub_img_Test, filename)
                shutil.copyfile(filepath, dst_img_file)

                dst_label_file = os.path.join(sub_label_Test, filename)
                shutil.copyfile(labelpath, dst_label_file)
            else:
                dst_img_file = os.path.join(sub_img_Train_Val, filename)
                shutil.copyfile(filepath, dst_img_file)

                dst_label_file = os.path.join(sub_label_Train_Val, filename)
                shutil.copyfile(labelpath, dst_label_file)



if __name__ == "__main__":

    current_gpu = '1'

    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))

    dataset_name = "Task04_Hippocampus"  #"Task01_BrainTumour" , "Task04_Hippocampus"
    print("dataset:", dataset_name)

    main(dataset_name)