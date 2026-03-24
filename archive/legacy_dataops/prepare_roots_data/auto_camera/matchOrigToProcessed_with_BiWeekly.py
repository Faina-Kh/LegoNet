import os
import sys
import shutil
import csv
from datetime import datetime, timedelta

################################################################################################################

# current_gpu = '0'
#
# os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
# print('Running on gpu {}'.format(current_gpu))

######################################################################################################################
# Fill in the following info

data_dir = os.path.join("F:\\Hatzeva 2024-2025\\Rootcam")

output_dir = os.path.join("F:\\Hatzeva 2024-2025\\Rootcam","processed_names", "Field2", "cam14") #"Bi-Weekly")
os.makedirs(output_dir, exist_ok=True)

######################################################################################################################

tube =1
cam_folders_info = {#"Cam2": {"folders":[*range(37, 53 +1), *range(64, 77 +1), *range(88, 99 +1), *range(110, 123 +1), *range(135, 160 +1),
                                        #*range(167, 194 +1),213, *range(222, 227 +1), *range(235, 241 +1)] , "num_of_Locations":32} #,[54, 78, 101, 124, 161, 197, 214]
                    #"Cam1": {"folders": [*range(66, 91 +1), *range(94, 109 +1), *range(122, 129 +1)], "num_of_Locations":32},
                    #"Cam4": {"folders": [*range(254, 271 +1), *range(273, 278 +1)], "num_of_Locations":32},
                    #"Cam4": {"folders": [272], "num_of_Locations": 25}
                  #"Cam5": {"folders": [*range(29, 145 +1), *range(149, 163 +1)], "num_of_Locations":41},
                  #"Cam10": {"folders": [*range(207, 228 +1), *range(230, 375 +1)], "num_of_Locations":24},
                 #"Cam3": {"folders": [*range(34, 50 +1)], "num_of_Locations":35},
                #"Cam6": {"folders": [*range(6, 37 +1)], "num_of_Locations":30}
                #"Cam9": {"folders": [*range(220, 261 +1), *range(264, 371 +1)], "num_of_Locations":23}
                #"Cam13": {"folders": [*range(93, 95+1), *range(98, 109 +1), *range(113, 117 +1),
                                     # *range(120, 157 +1), *range(159, 169 +1) ], "num_of_Locations":8}
                # "Cam13": {"folders": [*range(185, 212 +1), *range(217, 243+1)], "num_of_Locations":11}
               "Cam14": {"folders": [*range(119, 120 +1), 123, 126, 127], "num_of_Locations":9},
             "Cam14": {"folders": [128, 130, 133, 135, 139, 141, 143], "num_of_Locations":11}
                     }

# insert all folder numbers that you currently have for this tube
#folders_numbers = [*range(205,219+1)] #[*range(30, 36 +1)] # tube 9: [*range(66,135+1)], *range(205,219+1)] # example [*range(30, 36 +1), 50,*range(205,219+1)]]
#print("Folder numbers:", folders_numbers)

#num_of_Locations = 21

daily_imaging = True # False for bi-weekly

first_creation_of_tube_folder = True

if not first_creation_of_tube_folder:
    prev_last_date = "2025-01-05" #"2025-01-28"
    current_last_sess = 84 #51

########################################################################################################################

# confirm = input("Are the folders numbers correct? (yes/no): ").strip().lower()
# if confirm == "yes":
#     print("ok, let's continue..")
# else:
#     print("Operation canceled.")
#     sys.exit()  # Stop execution

########################################################################################################################

processesd_img_path = os.path.join(output_dir, "Tube_"+str(tube))
os.makedirs(processesd_img_path,exist_ok=True)

all_dates_for_tube = []
images_info = {}


for cam_num in cam_folders_info.keys():
    folders_numbers = cam_folders_info[cam_num]["folders"]
    print("cam_num:",cam_num)
    print("Folder numbers:", folders_numbers)

    ########################################################################################################################

    confirm = input("Are the folders numbers correct? (yes/no): ").strip().lower()
    if confirm == "yes":
        print("ok, let's continue..")
    else:
        print("Operation canceled.")
        sys.exit()  # Stop execution

    ########################################################################################################################
    prev_folder = "0"
    for f_num in folders_numbers:

        if f_num <100:
            folder="0"+str(f_num)
        else:
            folder = str(f_num)
        print("start working on folder:", folder, "\n")

        original_img_path = os.path.join(data_dir, cam_num, str(folder)) #cam_folders)

        # check if the folder exist
        if not os.path.exists(original_img_path):
            continue

        dir_list = os.listdir(original_img_path)

        if "L00" in dir_list:
            include_L = True
            original_img_path = os.path.join(original_img_path, "L00")
        else:
            include_L = False

        First_time_date = False
        for im in os.listdir(original_img_path):
            if im.lower().endswith(".jpg"):
                if ("_") in im:
                    splited_im = im.split("_")
                    current_img_num = str(int(splited_im[0].split("img")[1]))
                    current_date = splited_im[2]
                else:
                    splited_im = im.split(".jpg")[0]
                    current_img_num = str(int(splited_im[2:5]))
                    current_day = splited_im[8:10]
                    current_month = splited_im[10:12]
                    current_year = splited_im[12:16]
                    current_date = current_year+"-"+current_month+"-"+current_day


                if "2019-" in current_date:
                    if len(all_dates_for_tube) == 0:
                        raise TypeError("2019 images with no prevoiues dates")
                    else:
                        sorted_dates = sorted(all_dates_for_tube, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
                        last_date = sorted_dates[-1]
                        last_folder_num = images_info[last_date]["folder_num"]
                        if f_num == last_folder_num + 1:
                            current_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

                if current_date not in images_info.keys():
                    images_info[current_date] = {}
                    First_time_date = True
                elif not First_time_date:
                    raise TypeError("We already have this date: "+current_date, "Current folder:"+folder, "Prev folder: "+prev_folder, "cam: "+cam_num)


                if current_img_num in images_info[current_date].keys():
                    print(f"cam_num={cam_num}, f_num={f_num}, im={im}")
                    #raise TypeError(current_img_num, "is already in", current_date)
                    print(current_img_num, "is already in", current_date, "\n")
                    # if there are two images with num 2
                    if current_img_num =="2":
                        temp_dict = images_info[current_date]['2']
                        images_info[current_date]['1'] = temp_dict
                        del images_info[current_date]['2']

                images_info[current_date]["folder_num"] = f_num
                images_info[current_date]["cam_num"] = cam_num
                images_info[current_date][current_img_num] = {}
                images_info[current_date][current_img_num]["im_name"] = im

                if include_L:
                    images_info[current_date][current_img_num]["orig_path"] = os.path.join(cam_num, str(folder), "L00")
                else:
                    images_info[current_date][current_img_num]["orig_path"] = os.path.join(cam_num, str(folder))

                images_info[current_date][current_img_num]["date"] = current_date
                #images_info[current_date][current_img_num]["time"] = splited_im[3].split(".jpg")[0]

                if current_date not in all_dates_for_tube:
                    all_dates_for_tube.append(images_info[current_date][current_img_num]["date"])

        if os.listdir(original_img_path):
            prev_folder = folder

# sort the dates in chronological order to get session number
sorted_dates = sorted(all_dates_for_tube, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
dates_dict = {}

current_first_date = sorted_dates[0]
date1 = datetime.strptime(current_first_date, "%Y-%m-%d")

if first_creation_of_tube_folder:
    sess =1
else:
    if daily_imaging:
        # Convert strings to datetime objects
        date2 = datetime.strptime(prev_last_date, "%Y-%m-%d")

        # Calculate the difference in days
        days_between = abs((date2 - date1).days)

        if days_between<0:
            raise TypeError("Current first date "+current_first_date, "is earlier than prev last date "+prev_last_date)

        sess = current_last_sess+days_between
    else:
        sess = current_last_sess + 1

# arrange session numbers by dates for daily imaging data
for i in range(len(sorted_dates)):
    current_date = sorted_dates[i]
    if i==0:
        dates_dict[current_date] = sess
    else:
        if daily_imaging:
            date2 = datetime.strptime(current_date, "%Y-%m-%d")
            days_between = abs((date2 - date1).days)
            sess += days_between
            dates_dict[current_date] = sess
            date1 = date2
        else:
            sess += 1
            dates_dict[current_date] = sess


last_date = sorted_dates[-1]
last_sess = dates_dict[last_date]

# rename the images and copy with the new name. save the info to a csv.

print ("Start copying the images")
images_match_rows = []
for date in images_info.keys():

    if "1" in images_info[date].keys():
        start_from_one = True
        first_img_num = 1
    else:
        start_from_one = False
        first_img_num = 2

    cam_num = images_info[date]["cam_num"]
    for img_num in range(first_img_num, first_img_num + cam_folders_info[cam_num]["num_of_Locations"]):
        if start_from_one:
            Loc = cam_folders_info[cam_num]["num_of_Locations"] - img_num + 1
        else:
            Loc = cam_folders_info[cam_num]["num_of_Locations"] - img_num + 2

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
        #print(orig_path)

        dst_img_path = os.path.join(processesd_img_path, new_name)
        shutil.copy2(orig_path, dst_img_path)

        # add info to csv
        images_match_rows.append([date, current_sess, images_info[date][str(img_num)]["im_name"],new_name, images_info[date][str(img_num)]["orig_path"]])


all_cams = ""
for cam in cam_folders_info.keys():
    all_cams+="_"+cam

output_csv_name = "Tube_" + str(tube)+ all_cams+"_lastDate_"+last_date+ "_sess_"+str(last_sess) +"_images_match.csv"
images_match_file = os.path.join(processesd_img_path, output_csv_name)

# create the csv files
with open(images_match_file, "w", newline="") as file:
    writer = csv.writer(file)
    # Writing the header
    writer.writerow(["Date","Session", "Orig_name", "New_name", "Orig_im_path"])
    for row in images_match_rows:
        writer.writerow(row)
print("Done")
