import json
import os
import csv

########################################################################################################################

k="5"

data_path = os.path.join("E:\\roots_project", "Grapes_K_fold", "K_"+k)
                        #("C:\\Users\\Aragorn\\Desktop\\roots project", "Grapevine_data_all", "Tube 22_renamed")
                         #"July_22\\all_manual_2")
                         #"newAutoCam_17_7_22\\original_28_7_22")
                         #"Grapevine_data_all")  #"Rootfly_subfolders_Grapevine roots")
#"auto cam test_11_8_22\\raw_images") #"C:\\Users\\Aragorn\\Desktop\\roots project\\newAutoCam_17_7_22\\original_28_7_22"
# "July_22\\all_manual_2"

sub_dir = "Test"
txt_file = os.path.join(data_path, "sub_"+sub_dir+"_"+k, sub_dir+"_Dia_Length_Color.txt") #"all_data_Diameter_Length_Color.txt") #"sub_"+sub_dir, sub_dir+"_Dia_Length_Color.txt") # "Train_Dia_Length_Color_limit_5.txt") #'sub_Val', "Val_manual_Dia_and_Length.txt")

all_files = True
has_color = True

output_dia_mean = os.path.join(data_path, "sub_"+sub_dir+"_"+k, "dia_mean.csv") # "sub_"+sub_dir, sub_dir+"_dia_mean_limit_5.csv") #"dia_mean.csv"
output_dia_std = "" # os.path.join(data_path, "sub_"+sub_dir, sub_dir+"_dia_std_limit_5.csv") #"dia_std.csv") #,
output_roots_count = "" # = os.path.join(data_path, "sub_"+sub_dir, sub_dir+"_count_limit_5.csv") #"count.csv") #,

per_root_info = os.path.join(data_path, "sub_"+sub_dir+"_"+k, "per_root.csv") #"sub_"+sub_dir,sub_dir+"_per_root.csv" ) #"Test_per_root_limit_5.csv") #"count.csv") #,

name_type = "processed_name" #"processed_name" #"original_name"

########################################################################################################################

# reading the data from the file
with open(txt_file) as f:
    data = f.read()

roots_dict = json.loads(data)

color_dict = []

f = open(per_root_info, 'w', newline='')
with f:
    writer = csv.writer(f)
    if has_color:
        myrow = ["name", "root", "length", "dia", "color"]
    else:
        myrow = ["name", "root", "length", "dia"]
    writer.writerow(myrow)
    for im in roots_dict.keys():
        # create the row for the output file
        num = roots_dict[im]['roots_num']
        if num > 0:
            for key in roots_dict[im].keys():
                if "root_" in key:
                    myrow = []
                    myrow.append(roots_dict[im][name_type])
                    myrow.append(key)
                    myrow.append(roots_dict[im][key]['Root_Length'])
                    myrow.append(roots_dict[im][key]['Root_Diameter'])
                    if has_color:
                        if roots_dict[im][key]['Root_Color'] == "White":
                            myrow.append(1)
                        else:
                            myrow.append(0)
                    writer.writerow(myrow)

        else:
            myrow = []
            myrow.append(roots_dict[im][name_type])
            myrow.append(0)
            myrow.append(0)
            myrow.append(0)
            if has_color:
                myrow.append(-1)
            writer.writerow(myrow)



if all_files:

    f = open(output_dia_mean, 'w', newline='')
    with f:
        writer = csv.writer(f)
        for im in roots_dict.keys():
            # create the row for the output file
            myrow = []
            myrow.append(roots_dict[im][name_type])
            myrow.append(roots_dict[im]['roots_dia_mean'])
            writer.writerow(myrow)


    # f = open(output_dia_std, 'w', newline='')
    # with f:
    #     writer = csv.writer(f)
    #     for im in roots_dict.keys():
    #         # create the row for the output file
    #         myrow = []
    #         myrow.append(roots_dict[im][name_type])
    #         myrow.append(roots_dict[im]['roots_dia_std'])
    #         writer.writerow(myrow)
    #
    #
    # f = open(output_roots_count, 'w', newline='')
    # with f:
    #     writer = csv.writer(f)
    #     for im in roots_dict.keys():
    #         # create the row for the output file
    #         myrow = []
    #         myrow.append(roots_dict[im][name_type])
    #         myrow.append(roots_dict[im]['roots_num'])
    #         writer.writerow(myrow)
    #

