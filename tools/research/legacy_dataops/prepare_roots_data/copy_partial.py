import os
import csv
import shutil



#split the images in the RGB folder such that the corresponding RGB and GFP images will be in the same set

RGB_dir = os.path.join("D:\\Faina\\Roots\\Xu\\with_anns\\renamed_random_jpg_b", "all_images_with_anns", "RGB") #("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Tube 20_renamed")

GFP_dir = os.path.join("D:\\Faina\\Roots\\Xu\\with_anns\\renamed_random_jpg_b", "all_images_with_anns", "GFP")

RGB_num = "1"

# input paths
input_TRL = os.path.join(RGB_dir, "RGB_TRL.csv")
input_points = os.path.join(RGB_dir, "RGB_pointsOutput.csv")

input_TRL_dict = {}
input_points_dict = {}

with open(input_TRL, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        input_TRL_dict[row[0]]=row

with open(input_points, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        input_points_dict[row[0]]= [row[0]]
        for i in range(1, len(row)):
            if row[i]!="":
                input_points_dict[row[0]].append(row[i])



for dir in ["Train", "Val", "Test"]:
    copy_to = os.path.join(RGB_dir, "splitted", "sub_"+dir) #("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Additional image for testing", "Treatment 2")
    os.makedirs(copy_to, exist_ok=True)

    output_TRL_path = os.path.join(RGB_dir, "splitted", "sub_"+dir, dir+".csv")
    output_points_path = os.path.join(RGB_dir, "splitted", "sub_"+dir, dir+"_pointsOutput.csv")

    current_GFP_TRL_path = os.path.join(GFP_dir, "splitted", "sub_"+dir, dir+".csv")

    current_RGB_TRL = []
    current_RGB_points = []

    with open(current_GFP_TRL_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            spliited_GFP_name = row[0].split("-")
            Loc = spliited_GFP_name[3].split("_")[1]

            RGB_name = spliited_GFP_name[0] + "-" + spliited_GFP_name[1] + "-" + spliited_GFP_name[2] + "-" +RGB_num+"_"+ Loc

            # copy RGB image to relevant dir
            shutil.copyfile(os.path.join(RGB_dir, RGB_name), os.path.join(copy_to, RGB_name))

            current_RGB_TRL.append(input_TRL_dict[RGB_name])
            current_RGB_points.append(input_points_dict[RGB_name])

    dir_RGB_TRL = open(output_TRL_path, 'w', newline='')
    with dir_RGB_TRL:
        writer = csv.writer(dir_RGB_TRL)
        for row in current_RGB_TRL:
            writer.writerow(row)

    dir_RGB_points = open(output_points_path, 'w', newline='')
    with dir_RGB_points:
        writer = csv.writer(dir_RGB_points)
        for row in current_RGB_points:
            writer.writerow(row)