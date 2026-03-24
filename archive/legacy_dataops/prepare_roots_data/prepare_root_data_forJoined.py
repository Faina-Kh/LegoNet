import csv
import os
import cv2
import json

##################################################################################

generate_csv = False
generate_json = True

copy_images = False

##################################################################################

# Define paths

dataset_names = ["Tomato 2019", "Tomato 2020", "Pepper 2021", "corn 2020", "Melon 2019", "Melon 2018"]
data_path = os.path.join("D:/Faina/roots_project", "manual_camera", "13_5_22\\Correction after visual inspection_20220513")

output_csv_file = os.path.join(data_path, "manual_corrected.csv")
output_points_file = os.path.join(data_path, "manual_corrected_pointsOutput.csv")

save_cropped = True

with_time_format = True # True or False

##################################################################################

# output files
out_Train = os.path.join(data_path, "all_manual", "sub_Train")
out_Val = os.path.join(data_path, "all_manual", "sub_Val")
out_Test = os.path.join(data_path, "all_manual", "sub_Test")

os.makedirs(out_Train, exist_ok=True)
os.makedirs(out_Val, exist_ok=True)
os.makedirs(out_Test, exist_ok=True)

Train_TRL_file = os.path.join(data_path, out_Train, "Train.csv")
Val_TRL_file = os.path.join(data_path, out_Val, "Val.csv")
Test_TRL_file = os.path.join(data_path, out_Test, "Test.csv")

Train_pointsOutput_file = os.path.join(data_path, out_Train, "Train_pointsOutput.csv")
Val_pointsOutput_file = os.path.join(data_path, out_Val, "Val_pointsOutput.csv")
Test_pointsOutput_file = os.path.join(data_path, out_Test, "Test_pointsOutput.csv")

sub_options = ["Train", "Val", "Test"]


##################################################################################

for sub in sub_options:

    all_TRL = []
    all_points = []

    new_roots_dict = {}
    target_roots_file = os.path.join(data_path, "all_manual", "sub_"+sub, sub+"_manual_Dia_and_Length.txt")

    for dataset_name in dataset_names:

        current_path = os.path.join(data_path, dataset_name)

        # get roots data in txt files
        roots_input = os.path.join(current_path, "sub_"+sub, sub+"_Dia_and_Length.txt")

        with open(roots_input) as f:
            data = f.read()

        roots_dict = json.loads(data)

        for key in roots_dict:
            img_name = roots_dict[key]["processed_name"]
            if not os.path.exists(os.path.join(current_path, dataset_name + "_draw_GT", "dont use", "points_" + img_name)):
                new_roots_dict[dataset_name+"_"+key] = roots_dict[key]
            else:
                a=1


        # get TRL and points data
        raw_files = [dataset_name+ ".csv"]
        img_path = os.path.join(data_path, "images", dataset_name)

        # sub_Train = os.path.join(current_path, "sub_Train")
        # sub_Val = os.path.join(current_path, "sub_Val")
        # sub_Test = os.path.join(current_path, "sub_Test")

        current_dir = os.path.join(current_path, "sub_"+ sub)

        current_TRL_file = os.path.join(current_dir, sub+".csv")
        current_points_file = os.path.join(current_dir, sub + "_pointsOutput.csv")

        with open(current_TRL_file, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                img_name = row[0]

                if not os.path.exists(os.path.join(current_path,dataset_name+"_draw_GT", "dont use", "points_"+img_name)):
                    all_TRL.append(row)

                    if copy_images:
                        image = cv2.imread(os.path.join(img_path, img_name))
                        if save_cropped:
                            image = image[21:471, 16:, :]

                        dst_img_file = os.path.join(vars()["out_"+sub], img_name)
                        cv2.imwrite(dst_img_file, image)

        with open(current_points_file, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                img_name = row[0]

                if not os.path.exists(os.path.join(current_path,dataset_name+"_draw_GT","dont use", "points_"+img_name)):
                    all_points.append(row)



    # print to files

    if generate_csv:

        TRL_file = vars()[sub+"_TRL_file"]
        pointsOutput_file = vars()[sub+"_pointsOutput_file"]

        file = open(TRL_file, 'w', newline='')
        with file:
            writer = csv.writer(file)
            for row in all_TRL:
                writer.writerow(row)

        file = open(pointsOutput_file, 'w', newline='')
        with file:
            writer = csv.writer(file)
            for row in all_points:
                writer.writerow(row)


        if sub =="Train":

            file = open(Train_TRL_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_TRL:
                    writer.writerow(row)

            file = open(Train_pointsOutput_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_points:
                    writer.writerow(row)

        elif sub == "Val":

            file = open(Val_TRL_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_TRL:
                    writer.writerow(row)

            file = open(Val_pointsOutput_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_points:
                    writer.writerow(row)


        elif sub == "Test":

            file = open(Test_TRL_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_TRL:
                    writer.writerow(row)

            file = open(Test_pointsOutput_file, 'w', newline='')
            with file:
                writer = csv.writer(file)
                for row in all_points:
                    writer.writerow(row)

    if generate_json:
        with open(target_roots_file, "w") as outfile:
            json.dump(new_roots_dict, outfile)



print("Done")
