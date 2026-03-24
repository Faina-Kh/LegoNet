import os
import csv
from os.path import exists



########################################################################################################################

def main(dataset_name):

    data_dir = os.path.join("D:\\Faina\\roots_project", dataset_name) # "../data/"+ dataset_name
    original_img_path = os.path.join(data_dir, "cam3_original images")
    relevant_img_file = os.path.join(data_dir, "June 22", "TRL_June.csv") #"myFiles\\TRL_cam3.csv")

    images_match_file = os.path.join(data_dir, "June 22", "images_match_June.csv")

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

    processed_to_orig = {}

    with open(images_match_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            p = rows[0]
            o = rows[1].split("\\")[1]
            l = rows[2]

            processed_to_orig[p] = [o, l]

    with open(relevant_img_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img_name = row[0]
            orig_name = processed_to_orig[img_name][0]

            new_row = [orig_name, row[1]]

            if exists(os.path.join(Train_dir, orig_name)):
                Train_rows.append(new_row)
            elif exists(os.path.join(Val_dir, orig_name)):
                Val_rows.append(new_row)
            elif exists(os.path.join(Test_dir, orig_name)):
                Test_rows.append(new_row)




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