import os
import shutil
import numpy as np
import csv
import random



########################################################################################################################

def main(dataset_name):

    data_dir = os.path.join("D:\\Faina\\roots_project", dataset_name) # "../data/"+ dataset_name
    original_img_path = os.path.join(data_dir, "cam3_original images")
    relevant_img_file = os.path.join(data_dir, "June 22", "images_match_June.csv") #"myFiles\\TRL_cam3.csv")

    Train_dir = os.path.join(data_dir, "sub_Train")
    Val_dir = os.path.join(data_dir, "sub_Val")
    Test_dir = os.path.join(data_dir, "sub_Test")

    # create folders if don't exist
    os.makedirs(Train_dir, exist_ok=True)
    os.makedirs(Val_dir, exist_ok=True)
    os.makedirs(Test_dir, exist_ok=True)

    # splitted files
    Train_csv_file = os.path.join(Train_dir, "Train_June.csv")
    Val_csv_file = os.path.join(Val_dir, "Val_June.csv")
    Test_csv_file = os.path.join(Test_dir, "Test_June.csv")

    Train_rows = []
    Val_rows = []
    Test_rows = []

    sess_dict = {"1": "006", "2": "011", "3": "016", "4": "021", "5": "026", "6": "031", "7": "036", "8": "041",
                 "9": "046", "10": "051", "11": "056", "12": "061", "13": "067", "14": "074", "15": "079", "16": "084",
                 "17": "089", "18": "094", "19": "099", "20": "104", "21": "109", "22": "114", "23": "119", "24": "124",
                 "25": "129", "26": "134"}

    num_images = 832
    np.random.seed(0)
    random.seed(0)

    count= 0

    with open(relevant_img_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
             sess = int(row[0].split("_")[4].split(".jpg")[0])
             dir_num = sess_dict[str(sess)]
             Loc = int(row[0].split("_")[2].split("L")[1])

             current_dir = os.path.join(original_img_path, dir_num)
             dir_list = os.listdir(current_dir)
             check = 0
             for im in dir_list:
                 current_img_num = int(im.split("_")[0].split("img")[1])
                 if current_img_num==1:
                     check = 1
                     img_num = 32 - Loc + 1
                     break

             if check == 0:
                 #first img is img002
                 img_num = 32 - Loc + 2


             for x in dir_list:
                 current_img_num = int(x.split("_")[0].split("img")[1])

                 if current_img_num==img_num:
                     count += 1
                     print("copy file", count, "out of", num_images)

                     # choose randomly were to assign the image
                     rnd = np.random.random()

                     # copy images
                     if rnd <= 0.7:
                        Train_rows.append([x, row[1]])
                        dst_img_file = os.path.join(Train_dir, x)
                        shutil.copyfile(os.path.join(current_dir, x), dst_img_file)
                     elif 0.7< rnd <= 0.8:
                        Val_rows.append([x, row[1]])
                        dst_img_file = os.path.join(Val_dir, x)
                        shutil.copyfile(os.path.join(current_dir, x), dst_img_file)
                     else:
                        Test_rows.append([x, row[1]])
                        dst_img_file = os.path.join(Test_dir, x)
                        shutil.copyfile(os.path.join(current_dir,x), dst_img_file)

                     break



    f_Train = open(Train_csv_file, 'w', newline='')
    with f_Train:
        writer = csv.writer(f_Train)
        for row in Train_rows:
            writer.writerow(row)

    f_Val = open(Val_csv_file, 'w', newline='')
    with f_Val:
        writer = csv.writer(f_Val)
        for row in Val_rows:
            writer.writerow(row)


    f_Test = open(Test_csv_file, 'w', newline='')
    with f_Test:
        writer = csv.writer(f_Test)
        for row in Test_rows:
            writer.writerow(row)



if __name__ == "__main__":

    current_gpu = '1'

    os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
    print('Running on gpu {}'.format(current_gpu))

    dataset_name = "Rootfly_cam3"
    print("dataset:", dataset_name)

    main(dataset_name)