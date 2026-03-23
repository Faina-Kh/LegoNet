import csv
import os
import pandas as pd
import json

data_path = os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project", "Grapevine_data_all", "Tube 20_renamed")

input_file = os.path.join(data_path, "pred_Tube 20_for HM.csv")
output_file = os.path.join(data_path, "pred_Tube 20_for HM_out.csv")

range_count=10
count = 0
sum=0

first_row = True
my_dict = {}

max_range= 130 #130   #190 #(max_loc /7 *10)
max_Loc= 86 #128 #86

with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if first_row:
            my_dict["Loc_sess"] = [row[1], row[2], row[3], "Range", "Avg_white_Pct"]
            first_row=False
        else:
            if row[4]=="":
                sum += 0
            else:
                sum+=float(row[4]) #gt white %
            count+=1

            if count==7 or int(row[2])==max_Loc:

                my_dict["Loc"+row[2]+"_sess"+row[3]] = [row[1], row[2], row[3], -1*range_count]
                if count==7:
                    my_dict["Loc"+row[2]+"_sess"+row[3]].append(sum/7)
                else:
                    my_dict["Loc" + row[2] + "_sess" + row[3]].append(sum/count)

                sum=0
                count=0
                if range_count==max_range:
                    range_count = 10
                else:
                    range_count+=10



f = open(output_file, 'w', newline='')
with f:
    writer = csv.writer(f)
    for k in my_dict.keys():
        #create the row for the output file
        myrow = []
        myrow.append(k)
        myrow.append(my_dict[k][0])
        myrow.append(my_dict[k][1])
        myrow.append(my_dict[k][2])
        myrow.append(my_dict[k][3])
        myrow.append(my_dict[k][4])
        writer.writerow(myrow)