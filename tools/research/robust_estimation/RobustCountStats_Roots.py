import os

current_gpu = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GaussAndUniformEM import GaussAndUniformEM, TwoGaussiansEM
from evaluateEstimation import evaluateEstimation


use_beta = True

#beta=0.5
#min_conf='0.9'
#field='1'
EMfunc = TwoGaussiansEM #GaussAndUniformEM #TwoGaussiansEM
line=0

all_conf= ["0.05","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"] #["0.01",
all_beta= [0.2,0.5,0.7,1]

one_field = False
csv_columns = ["beta", "min conf", "method","MAE", "MSE", "error", "1-FVU"] #"field",
break_loops = False
###########################################################################################################################

relevant_out= "avg" # "sum", "avg"
include_empty_images = False

per_root_pheno = "color" #"length" # "dia", "color"
phenoytpe = "color" #"color" #"dia" #"TRL"

gt_set= "Test"

write_row = True

raw_pred_file = os.path.join("both_2_detect_3Sets\\reg\\I0.5_s0.7_10_dia_100_color\\trained_all", "Test_I_0.5_s_0.7_222\\")  #"val_I 0.5_s 0.7_all_epoch229\\"
#os.path.join("eval_mixed_weights", "find2_b_backbone_2_b_I_0.5_s_0.7_Test_2\\") # #"find2_b_backbone_2_b_I_0.5_s_0.7_Test_2\\") #"find2_b_backbone_2_bVal_I 0.5_s 0.7\\")
DataPath = os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project\\Grapevine_data_all\\Results",raw_pred_file) #"find2_b_backbone_2_b_I_0.5_s_0.7_Test_2\\") #"find2_b_backbone_2_bVal_I 0.5_s 0.7\\")

# This file contains a matrix 115*5. Each row is a detection
#   Each detection is described by {'predicted count'  'gt_count'  'label'  'score'  'imageind'}

detect_file= "detections_data_any_crop_withEmptyIm.csv"  #"detections_data_any_crop.csv"  #"detections_data_any_crop_withEmptyIm.csv" # - Copy.csv"
not_found_file="not_found_gt_count.csv" #"not_found_gt_count_2.csv"    #"not_found_gt_count.csv"

############################################################################################################################
val_pred_path = os.path.join("C:\\Users\\Aragorn\\Desktop", "roots project\\Grapevine_data_all\\Results",
                             "both_2_detect_3Sets\\reg\\I0.5_s0.7_10_dia_100_color\\trained_all\\val_I 0.5_s 0.7_all_epoch229\\")+"detections_data_any_crop.csv"

                        #"eval_mixed_weights\\find2_b_backbone_2_bVal_I 0.5_s 0.7\\")+"detections_data_any_crop.csv"

###########################################################################################################################


results_file = DataPath+ "robust_results_"+phenoytpe+"_"+gt_set+"_all.csv"

output_file =  DataPath+ "robust_outputs_"+phenoytpe+"_"+gt_set+"_all.csv"


###########################################################################################################################

val_pred_file = pd.read_csv(val_pred_path)



for min_conf in all_conf:
    for beta in all_beta:

        #DataPath= os.path.join("C:\\Users\\Aragorn\\Desktop","Robust new", "3.try again", 'conf '+ min_conf+'\\')
                               #"roots project", "Grapevine_data_all", "Results", "both_2_detect_3Sets\\reg", "I0.5_s0.7_10_dia_100_color\\trained_all",
                               #"detections_data_any_crop_withEmptyIm.csv")
                               #current_dir, "again_test", 'conf ' + min_conf + '\\')
                               #"val set", 'conf '+ min_conf+'\\') #"test initial data"
                               # "val fields\\plot 135", 'conf '+ min_conf+'\\')
                               # "3.try again", field, 'conf '+ min_conf+'\\')
                                                             # #wheat_test robust\\val set",'conf '+ min_conf+'\\')
                                                            #"Robust new", field,,'conf '+ min_conf+'\\')
        break_loops = False
        if not os.path.exists(DataPath+detect_file):
            break_loops=True
            break

        all_detections = pd.read_csv(DataPath + detect_file, usecols=["img", "pred_"+phenoytpe, "gt_"+phenoytpe, "label", "score"]) # header=0, [0, 1, 2, 3, 4]) #pd.read_csv(DataPath+"detections_data_any_crop_score 0.05_processed4.csv",header=None,usecols=[0,1,2,3,4])
        Undetected = pd.read_csv(DataPath + not_found_file, usecols=["img", "pred_"+phenoytpe, "gt_"+phenoytpe, "label", "score"])    # header=0,   header=0, [0, 1, 2, 3, 4])

        # move 'imageind' to the end
        cols = [col for col in all_detections if col != all_detections.columns[0]] + [all_detections.columns[0]]
        all_detections = all_detections[cols]      #Data3.columns=[ 'pred' , 'gt_count' , 'label' , 'score' , 'imageind']
        Undetected = Undetected[cols]

        #all_detections = all_detections.to_numpy() # true and false detections
        #Undetected = Undetected.to_numpy()

        if one_field:
            for i in range(all_detections.shape[0]):
                all_detections[i][4] = 'img'
            for i in range(Undetected.shape[0]):
                Undetected[i][4] = 'img'

        joined_data = pd.concat((all_detections, Undetected)) #np.concatenate((all_detections, Undetected))


        STDestimationFlag=0
        Alpha =0.5     # Alpha = P(h=1) = prior probability of the Gaussian in the EM formulation
        Temperature=1  # Temperature for confidence weights

        # Distribution of counts for detections and false alarms
        ShowDistributionFlag=0

        ##################################################################################
        # need to categorize to bins
        ##################################################################################

        if ShowDistributionFlag:
            F=plt.figure(1)
            colors=['b','r']
            Labels=[1,  0]
            Nbad=np.argwhere((all_detections["label"]==0).to_numpy()).size #np.argwhere(all_detections[:,2]==0).size # Number of False detections
            NGood=np.argwhere((all_detections["label"]==1).to_numpy()).size   #all_detections[:,2]==0  # number of good detections
            for i in range(2):
                Inds = np.argwhere((all_detections["label"] == Labels[i]).to_numpy()) #all_detections[:, 2] # indices of true and false detections (among the available detections)
                [Values, Counts] = np.unique((all_detections["pred_TRL"].to_numpy())[Inds], return_counts=True) #all_detections[Inds, 0]
                p=np.asarray([Values, Counts]).T    # P holds the distribution of predicted counts as a P*2 table. Each row is (predicted count value, how many times it occurs in the data)
                plt.plot(p[:,0],p[:,1],color=colors[i]) # The distributions of flase alarm counts (red) and true positive counts (blue)

            plt.show()

        # Here we try to estimate the mean and std of the count for images.
        # images are considered as sets of detected rectangles. We think of each image as describing a different field.
        # The mean and variance we want to estimate are of the spikelet count (per spike) in the field
        # Here field means are estimated from single image (each image is
        # considered a field), but the idea can be trivially extended to estimating mean and var from multiple images of the same field.


        if not include_empty_images:
            if per_root_pheno != "color":
                idx = (np.argwhere(joined_data["gt_" + phenoytpe].to_numpy() >0))
            else:
                idx = np.argwhere(joined_data["gt_" + phenoytpe].to_numpy() != -1)
            joined_data = joined_data[["gt_"+phenoytpe, "pred_"+phenoytpe, "label", "score", "img"]].to_numpy()[idx.squeeze(1)]
            joined_data = pd.DataFrame(joined_data, columns=["gt_"+phenoytpe, "pred_"+phenoytpe, "label", "score", "img"])



        ImageNames = np.unique(joined_data["img"])  # joined_data[:, 4] # list of image indices
        ImagesNum=ImageNames.shape[0]

        # initialize all statistics
        MeanEstflag=np.zeros(ImagesNum, dtype=bool)
        MeanEstWithFAFlag=np.zeros(ImagesNum, dtype=bool)
        StdEstflag=np.zeros(ImagesNum, dtype=bool)
        StdEstWithFAFlag=np.zeros(ImagesNum, dtype=bool)

        oracleFlag = np.zeros(ImagesNum, dtype=bool)

        Ns=np.zeros(ImagesNum)
        TrueM=np.zeros(ImagesNum)     # GT mean
        TrueS=np.zeros(ImagesNum)     # GT std
        EstM=np.zeros(ImagesNum)      # plain estimation of mean
        EstS=np.zeros(ImagesNum)      # plain estimation of std
        EstMO=np.zeros(ImagesNum)     # mean estimation  with oracle
        EstSO=np.zeros(ImagesNum)     # std estimation with oracle
        EstM_EM=np.zeros(ImagesNum)   # Plain EM mean
        EstS_EM=np.zeros(ImagesNum)   # plain EM std
        EstMC=np.zeros(ImagesNum)     # Confidence based mean
        EstSC=np.zeros(ImagesNum)     # confidence based std
        EstMM=np.zeros(ImagesNum)     # median mean
        EstSM=np.zeros(ImagesNum)     # median based std - not used
        EstM_EMC=np.zeros(ImagesNum)  # EM with confidence - mean
        EstS_EMC=np.zeros(ImagesNum)  # EM with confidence - std
        EstMCM=np.zeros(ImagesNum)    # median with confidence weights
        EstSCM=np.zeros(ImagesNum)    # std from  median with confidence weights? - not used
        ImageSpikes=[None]*ImagesNum  # Keep the detection sets

        FA_pred = all_detections["pred_"+phenoytpe].to_numpy()[np.argwhere(all_detections["label"].to_numpy() == 0)].squeeze()
        all_pred_range = np.max(all_detections["pred_"+phenoytpe])-np.min(all_detections["pred_"+phenoytpe])

        #pred_range=np.max(FA_pred)-np.min(FA_pred)
        FA_mean = "" #np.mean(FA_pred) #10 #np.mean(FA_pred)
        FA_std = "" #np.std(FA_pred)  #4 #np.std(FA_pred)

        inds = np.argwhere(val_pred_file["label"].to_numpy() == 0 )
        FA_mean = np.mean(val_pred_file["pred_"+phenoytpe].to_numpy()[inds.squeeze(1)])
        FA_std = np.std(val_pred_file["pred_"+phenoytpe].to_numpy()[inds.squeeze(1)])

        #print("FA_mean", FA_mean)
        #print("FA_std", FA_std)

        #Compute the estimations, looping over samples
        for i in range(ImagesNum): # collect only 'interesting' images: those for which some spikes are detected, some don't
            #if ImageNames[i]=='T040_L053_2013.05.19_090054_004.jpg':
             #   a=1

            if i==16:
                a=1

            inds=np.argwhere(all_detections["img"].to_numpy() == ImageNames[i]) # find indices of all the detections belonging to image ImageNums(i)
            objects= all_detections[["pred_"+phenoytpe, "gt_"+phenoytpe, "label", "score"]].to_numpy()[inds.squeeze(1)]#, 0:4] # get the spikes detections of ImageNums(i)
            N=objects.shape[0]
            TrueInds= np.argwhere(objects[:,2]==1) # indices of real objects
            ###################################################################################################
            MeanEstflag[i] =  True #TrueInds.shape[0]>0 # Should this field  be used for mean estimation?
            ####################################################################################################

            MeanEstWithFAFlag[i]=  True #not (TrueInds.shape[0]==N or TrueInds.shape[0]==0)  # Should this field  be used for std estimatuon with FAs?
            StdEstflag[i] =  TrueInds.shape[0]>1 and  np.std(objects[TrueInds,1])>0  #  Should this field  be used for std estimatuon?
                                    # only if there are at least two real samples and thery are different (i.e. the real std is >0)
            StdEstWithFAFlag[i]= StdEstflag[i] and not(TrueInds.shape[0]==N or TrueInds.shape[0]<=1) # Should this field  be used for std estimatuon with FAs?

            # store the image example
            Ns[i]=N # number of object detections in the image

            # get all gt data for current img
            img_gt_inds = np.argwhere(np.multiply((joined_data["img"]==ImageNames[i]).to_numpy(), (joined_data["label"]==1).to_numpy()))
            trueCounts = joined_data["gt_"+phenoytpe].to_numpy()[img_gt_inds.squeeze(1)]

            if MeanEstflag[i]:
                # ground truth
                if relevant_out == "sum":
                    TrueM[i] = np.sum(trueCounts)
                elif relevant_out == "avg":
                    TrueM[i]=np.mean(trueCounts) #Spikes[TrueInds,1]) # true mean (better be estimated also by adding true undeteced spikes)

                TrueS[i]=np.std(trueCounts)  #Spikes[TrueInds,1]) # true std

                # simple estimation
                if relevant_out == "sum":
                    EstM[i]=np.sum(objects[:,0]) # sum of network predictions
                elif relevant_out == "avg":
                    EstM[i]=np.mean(objects[:,0])  # mean of network predictions

                EstS[i]=np.std(objects[:,0])   # std of network predictions

                # Estimation with a false alarm oracle (not practical - just to provide a bound over the best we can do

                if len(TrueInds)>0:
                    oracleFlag[i]=True

                #To-Do fix the total estimation of an oracle with images with only FA

                if relevant_out == "sum":
                    EstMO[i] = np.sum(objects[TrueInds, 0])
                elif relevant_out == "avg":
                    EstMO[i]=np.mean(objects[TrueInds,0]) # mean as estimated by network prediction + an magical oracle of false positives
                # i,e, what could we have obtained if we had somehow known which detection is a false positive
                EstSO[i]=np.std(objects[TrueInds,0])  # std as estimated by network prediction + a magical oracle of false positives

                # Robust EM estimator
                P_UniformD= None #1/all_pred_range  #1/15 #1/pred_range

                Preds=np.array(objects[:,0], dtype=float) # prediction values to estimate the mean/sum from. These are visible values X to explain

                EstM_EM[i], EstS_EM[i] = EMfunc(Preds,P_UniformD, Alpha, FA_mean, FA_std, relevant_out) #GaussAndUniformEM(Preds,P_UniformD,Alpha) #TwoGaussiansEM

                # Estimating with Weighting by detector confidence
                orig_conf=objects[:,3]
                if use_beta:
                    conf= 1/(1+((1-orig_conf)/orig_conf)**beta)
                else:
                    conf =orig_conf

                # 'Confidence estimation'
                if relevant_out == "sum":
                    EstMC[i] = np.dot(conf.T, Preds)
                elif relevant_out == "avg":
                    EstMC[i] = np.dot(conf.T ,Preds)/np.sum(conf)

                EstSC[i] = np.sqrt( np.dot(conf.T, np.square( Preds-EstMC[i])) / np.sum(conf))

                # median
                if relevant_out == "sum":
                    EstMM[i] = np.nan
                elif relevant_out == "avg":
                    EstMM[i] = np.median(Preds)

                EstSM[i] = np.nan

                # 'EM_with_Confidence'
                #GaussProbs = Spikes[:, 3]
                GaussProbs = np.array(conf,dtype=float)
                GaussProbs = 1 / (1 + np.exp(- (np.log(GaussProbs/ (1 - GaussProbs)) ) / Temperature) )  # temperature
                # if temperature is high, we are less confident in our binary spike / non spike decisions

                EstM_EMC[i], EstS_EMC[i] = EMfunc(Preds, P_UniformD, GaussProbs, FA_mean, FA_std, relevant_out) #TwoGaussiansEM #GaussAndUniformEM(Preds, P_UniformD, GaussProbs)

                # median with confidence weights
                if relevant_out == "sum":
                    EstMCM[i]=np.nan
                elif relevant_out == "avg":
                    SortedPreds = np.sort(Preds)
                    arginds = np.argsort(Preds)
                    SortedGaussProbs = GaussProbs[arginds]
                    SecondHalf = np.argwhere(np.cumsum(SortedGaussProbs) / np.sum(SortedGaussProbs) > 0.5)
                    ind = SecondHalf[0]
                    if ind == 0:
                        EstMCM[i] = SortedPreds[0]
                    else:
                        EstMCM[i] = (SortedPreds[ind - 1] * SortedGaussProbs[ind - 1] + SortedPreds[ind] *
                                     SortedGaussProbs[ind]) / \
                                    (SortedGaussProbs[ind - 1] + SortedGaussProbs[ind])
                        # EstMCM[i] = SortedPreds[ind-1];

                EstSCM =np.nan

                ImageSpikes[i]=objects

            else:
                a=1

        # Mean estimation - only with false alarms
        MethodKeys=[ 'Standard estimation', 'FA oracle estimtion' , 'EM estimation' , 'Confidence estimation' , 'Median' , \
            'EM_with_Confidence' , 'Confidence Median']


        print(MethodKeys)


        MeasurementKeys=['Mean absolute error' , 'MSE' , 'Relative Error', '1-FUV' ]
        Estimators= np.vstack( [ EstM[MeanEstWithFAFlag] , EstMO[MeanEstWithFAFlag] , EstM_EM[MeanEstWithFAFlag] , \
                                 EstMC[MeanEstWithFAFlag] , EstMM[MeanEstWithFAFlag] , EstM_EMC[MeanEstWithFAFlag], \
                                 EstMCM[MeanEstWithFAFlag]  ])
        Mean_Accuracy_FA= evaluateEstimation( TrueM[MeanEstWithFAFlag],Estimators)

        print('min conf:{},  beta:{}'.format(min_conf, str(beta))) #print('field:{} , min conf:{},  beta:{}'.format(field, min_conf,str(beta)))
        #print("Mean_Accuracy_FA")
        #for i in range(len(Mean_Accuracy_FA)):
        #    print(Mean_Accuracy_FA[i])
        #print( 'N={}\n\n'.format(sum(MeanEstWithFAFlag)))

        # Mean estimation - general
        Estimators= np.vstack( [ EstM[MeanEstflag] , EstMO[MeanEstflag] , EstM_EM[MeanEstflag] , \
                                 EstMC[MeanEstflag] , EstMM[MeanEstflag] , EstM_EMC[MeanEstflag], \
            EstMCM[MeanEstflag]  ])



        Mean_Accuracy_General = evaluateEstimation(TrueM[MeanEstflag],Estimators)

        print("Mean_Accuracy_General")
        for i in range(len(Mean_Accuracy_General)):
            print(Mean_Accuracy_General[i])

        #print to csv


        f= open(output_file,  'a', newline='')
        with f:
            writer = csv.writer(f)
            if  write_row:
                writer.writerow(["min_conf", "beta", "gt",  'Standard estimation', 'FA oracle estimtion' , 'EM estimation' , 'Confidence estimation' , 'Median' , \
                'EM_with_Confidence' , 'Confidence Median'])
                write_row=False
            gt = TrueM[MeanEstflag]
            for i in range(Estimators.shape[1]):
                myrow = []
                myrow.append(min_conf)
                myrow.append(beta)
                myrow.append(gt[i])
                Estimator = Estimators[:,i]
                for j in range(7):
                    myrow.append(Estimator[j])
                writer.writerow(myrow)


        if 1:
            #export detections info
            f = open(results_file, 'a', newline='')
            with f:
                writer = csv.writer(f)
                if line==0:
                    writer.writerow(csv_columns)
                    line=1
                for j in range(len(MethodKeys)):
                    myrow = []
                    myrow.append(str(beta))
                    myrow.append(min_conf)
                    myrow.append(MethodKeys[j])
                    myrow.append(Mean_Accuracy_General[j][0])
                    myrow.append(Mean_Accuracy_General[j][1])
                    myrow.append(Mean_Accuracy_General[j][2])
                    myrow.append(Mean_Accuracy_General[j][3])
                    writer.writerow(myrow)


            print( 'N={}\n\n'.format(sum(MeanEstflag)))

            if STDestimationFlag: # Std estimation
                Estimators = np.vstack([EstM[StdEstWithFAFlag], EstMO[StdEstWithFAFlag], EstM_EM[StdEstWithFAFlag], \
                                        EstMC[StdEstWithFAFlag], EstMM[StdEstWithFAFlag], EstM_EMC[StdEstWithFAFlag], \
                                        EstMCM[StdEstWithFAFlag]])
                std_Accuracy= evaluateEstimation(TrueS[StdEstWithFAFlag],Estimators)
                print( 'N={}\n\n'.format(sum(StdEstWithFAFlag)))
                q=np.vstack( (TrueS[StdEstWithFAFlag] , Estimators ))
                np.corrcoef(q[:,2:].T)

