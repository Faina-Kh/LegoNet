import os
import sys
import shutil
import csv
from datetime import datetime

from torch.xpu import seed_all

################################################################################################################

current_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# Fill in the following info

data_dir = os.path.join("D:\\Faina\\Roots\\Sharon","all_images")

output_dir = os.path.join("D:\\Faina\\Roots\\Sharon","processed_names")
os.makedirs(output_dir, exist_ok=True)

######################################################################################################################

tube = 9
cam_folder = "Cam9"

# insert all folder numbers that you currently have for this tube
folders_numbers = [*range(205,219+1)] #[*range(30, 36 +1)] # tube 9: [*range(66,135+1)], *range(205,219+1)] # example [*range(30, 36 +1), 50,*range(205,219+1)]]
print("Folder numbers:", folders_numbers)

num_of_Locations = 21

daily_imaging = True

first_creation_of_tube_folder = False

if not first_creation_of_tube_folder:
    prev_last_date = "2025-01-28"
    current_last_sess = 51

########################################################################################################################

confirm = input("Are the folders numbers correct? (yes/no): ").strip().lower()
if confirm == "yes":
    print("ok, let's continue..")
else:
    print("Operation canceled.")
    sys.exit()  # Stop execution

########################################################################################################################

processesd_img_path = os.path.join(output_dir, "Tube_"+str(tube))
os.makedirs(processesd_img_path,exist_ok=True)


all_dates_for_tube = []
images_info = {}

for f_num in folders_numbers:
    if f_num <100:
        folder="0"+str(f_num)
    else:
        folder = str(f_num)

    original_img_path = os.path.join(data_dir, cam_folder, str(folder))

    # check if the folder exist
    if not os.path.exists(original_img_path):
        continue

    dir_list = os.listdir(original_img_path)

    if "L00" in dir_list:
        include_L = True
        original_img_path = os.path.join(original_img_path, "L00")
    else:
        include_L = False

    for im in os.listdir(original_img_path):
        if im.lower().endswith(".jpg"):
            splited_im = im.split("_")
            current_img_num = str(int(splited_im[0].split("img")[1]))
            current_date = splited_im[2]
            if current_date not in images_info.keys():
                images_info[current_date] = {}

            images_info[current_date][current_img_num] = {}
            images_info[current_date][current_img_num]["im_name"] = im

            if include_L:
                images_info[current_date][current_img_num]["orig_path"] = os.path.join(cam_folder, str(folder), "L00")
            else:
                images_info[current_date][current_img_num]["orig_path"] = os.path.join(cam_folder, str(folder))

            images_info[current_date][current_img_num]["date"] = current_date
            images_info[current_date][current_img_num]["time"] = splited_im[3].split(".jpg")[0]

            if current_date not in all_dates_for_tube:
                all_dates_for_tube.append(images_info[current_date][current_img_num]["date"])

# sort the dates in chronological order to get session number
sorted_dates = sorted(all_dates_for_tube, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
dates_dict = {}

current_first_date = sorted_dates[0]
date1 = datetime.strptime(current_first_date, "%Y-%m-%d")

if first_creation_of_tube_folder:
    sess =1
else:
    # Convert strings to datetime objects
    date2 = datetime.strptime(prev_last_date, "%Y-%m-%d")

    # Calculate the difference in days
    days_between = abs((date2 - date1).days)

    sess = current_last_sess+days_between

# arrange session numbers by dates for daily imaging data
for i in range(len(sorted_dates)):
    current_date = sorted_dates[i]
    if i==0:
        dates_dict[current_date] = sess
    else:
        date2 = datetime.strptime(current_date, "%Y-%m-%d")
        days_between = abs((date2 - date1).days)
        sess += days_between
        dates_dict[current_date] = sess
        date1 = date2

last_date = sorted_dates[-1]
last_sess = dates_dict[last_date]

# rename the images and copy with the new name. save the info to a csv.
images_match_rows = []
for date in images_info.keys():

    if "1" in images_info[date].keys():
        start_from_one = True
        first_img_num = 1
    else:
        start_from_one = False
        first_img_num = 2

    for img_num in range(first_img_num, first_img_num+num_of_Locations):
        if start_from_one:
            Loc = num_of_Locations - img_num + 1
        else:
            Loc = num_of_Locations - img_num + 2

        current_sess = dates_dict[date]

        if current_sess<10:
            sess_str = "00"+str(current_sess)
        elif current_sess<100:
            sess_str = "0" + str(current_sess)
        else:
            sess_str = str(current_sess)

        if tube<10:
            tube_str = "00"+str(tube)
        else:
            tube_str = "0" + str(tube)

        if Loc <10:
            Loc_str = "00" + str(Loc)
        else:
            Loc_str = "0" + str(Loc)

        new_name = "Pepper_T" + tube_str + "_L_" + Loc_str + "_" + date + "_" + sess_str + ".jpg"

        # copy the image with the new name
        orig_path = os.path.join(data_dir,images_info[date][str(img_num)]["orig_path"], images_info[date][str(img_num)]["im_name"])
        dst_img_path = os.path.join(processesd_img_path, new_name)
        shutil.copy2(orig_path, dst_img_path)

        # add info to csv
        images_match_rows.append([date, current_sess, images_info[date][str(img_num)]["im_name"],new_name, images_info[date][str(img_num)]["orig_path"]])



output_csv_name = "Tube_" + str(tube)+"_"+ cam_folder+"_lastDate_"+last_date+ "_sess_"+str(last_sess) +"_images_match.csv"
images_match_file = os.path.join(processesd_img_path, output_csv_name)

# create the csv files
with open(images_match_file, "w", newline="") as file:
    writer = csv.writer(file)
    # Writing the header
    writer.writerow(["Date","Session", "Orig_name", "New_name", "Orig_im_path"])
    for row in images_match_rows:
        writer.writerow(row)
print("Done")
