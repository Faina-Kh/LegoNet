import pandas as pd
import csv
import os


data_dir="C:\\Users\\Aragorn\\Google Drive\\StoragePath\\Datasets\\KK_datasets\\131_wheat_spikes_and_spikelets\\test_diff_fields\\try again"
csv_file = os.path.join(data_dir, "plot 128\\task 164\\128_164.csv") #"plot 1\\task 129\\129.csv") #"plot 128\\task 164\\128_164.csv")#"plot 1\\task 129\\129.csv") # task 125\\1111.csv")
test_file = os.path.join(data_dir, "plot 128\\task 164\\test_128_164.csv") #"plot 1\\task 129\\test_1_129.csv") #"plot 128\\task 164\\test_128_164.csv")#"plot 1\\task 129\\test_1_129.csv") #task 125\\test_1_125.csv")

#read_file = pd.read_csv(csv_file)

with open(csv_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    f = open(test_file, 'w', newline='')
    with f:
        writer = csv.writer(f)
        for row in csv_reader:
           myrow = []
           im=row[0]
           if row[1]=="Spike":
               if row[2]=="measureable":
                   x1=int(float(row[3]))
                   y1=int(float(row[4]))
                   w=int(float(row[5]))
                   h=int(float(row[6]))
                   x2=x1+w
                   y2=y1+h

                   myrow.append(im)
                   myrow.append('wheat')
                   myrow.append(x1)
                   myrow.append(y1)
                   myrow.append(x2)
                   myrow.append(y2)
                   writer.writerow(myrow)

           elif row[1]=="Spikelets":

               temp=row[7] # first xi
               count=0
               while temp!= "":
                   count+=1
                   temp=row[7+count]

               num_of_points=int(count/2)

               # for p in range(7, 7+count-1, 2):
               #     myrow = []
               #     myrow.append(im)
               #     myrow.append('wheat')
               #     myrow.append(int(float(row[p]))) #add xi
               #     myrow.append(int(float(row[p+1]))) #add yi
               #     writer.writerow(myrow)

               for p in range(7, 7+num_of_points, 1):
                   myrow = []
                   myrow.append(im)
                   myrow.append('wheat')
                   myrow.append(int(float(row[p]))) #add xi
                   myrow.append(int(float(row[p+num_of_points]))) #add yi
                   writer.writerow(myrow)



