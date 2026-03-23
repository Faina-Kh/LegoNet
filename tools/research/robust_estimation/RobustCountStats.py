import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIR = CURRENT_DIR
while not (os.path.exists(os.path.join(SEARCH_DIR, "config.py")) and os.path.isdir(os.path.join(SEARCH_DIR, "legonet"))):
    parent = os.path.dirname(SEARCH_DIR)
    if parent == SEARCH_DIR:
        break
    SEARCH_DIR = parent
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)

current_gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
print('Running on gpu {}'.format(current_gpu))


# storagePath
import paths
myStoragePath = paths.STORAGE_PATH
myDatasetsPath = paths.DATASETS_PATH
myExpResultsPath = paths.EXP_RESULTS_PATH
myModelsPath = paths.MODELS_PATH

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GaussAndUniformEM import GaussAndUniformEM, TwoGaussiansEM
from evaluateEstimation import evaluateEstimation


# This file contains a matrix 115*5. Each row is a detection
#   Each detection is described by {'predicted count'  'gt_count'  'label'  'score'  'imageind'}

detect_file= "detections_data_any_crop.csv" # - Copy.csv"
not_found_file="not_found_gt_count.csv" #"not_found_gt_count_2.csv"    #"not_found_gt_count.csv"
use_beta = True

#beta=0.5
#min_conf='0.9'
#field='1'
EMfunc = TwoGaussiansEM #GaussAndUniformEM #TwoGaussiansEM
line=0
all_fields= ["all val"] #["plot 1", "plot 109", "plot 114", "plot 115"  ,"plot 128" ,"plot 135"] #["1","2","3","4","5"]
all_conf= ["0.05","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"] #["0.01",
all_beta= [1] #[0.2,0.5,0.7,1]

one_field = False

csv_file = os.path.join("C:\\Users\\Aragorn\\Desktop","Robust new", "val_all_qa.csv")
#"3.try again", "test new fields.csv")
#"\\val fields\\plot 135\\results_val 135_one field.csv")
csv_columns = ["field", "beta", "min conf", "method","MAE", "MSE", "error", "1-FVU"]

break_loops = False
for field in all_fields:
    for min_conf in all_conf:
        for beta in all_beta:

            DataPath= os.path.join("C:\\Users\\Aragorn\\Desktop\\Robust new", "val set", 'conf '+ min_conf+'\\')
                                   # "val fields\\plot 135", 'conf '+ min_conf+'\\')
                                   # "3.try again", field, 'conf '+ min_conf+'\\')
                                                                 # #wheat_test robust\\val set",'conf '+ min_conf+'\\')
                                                                #"Robust new", field,,'conf '+ min_conf+'\\')
            break_loops = False
            if not os.path.exists(DataPath+detect_file):
                break_loops=True
                break

            Data3 = pd.read_csv(DataPath+detect_file,header=0,usecols=[0,1,2,3,4]) #pd.read_csv(DataPath+"detections_data_any_crop_score 0.05_processed4.csv",header=None,usecols=[0,1,2,3,4])

            # move 'imageind' to the end
            cols = [col for col in Data3 if col != Data3.columns[0]] + [Data3.columns[0]]
            Data3 = Data3[cols]

            #Data3.columns=[ 'pred' , 'gt_count' , 'label' , 'score' , 'imageind']

            m=Data3.to_numpy()

            Undetected = pd.read_csv(DataPath + not_found_file, header=0, usecols=[0,1,2,3,4])  # pd.read_csv(DataPath+"detections_data_any_crop_score 0.05_processed4.csv",header=None,usecols=[0,1,2,3,4])
            Undetected = Undetected.to_numpy()

            if one_field:
                for i in range(m.shape[0]):
                    m[i][4] = 'img'

                for i in range(Undetected.shape[0]):
                    Undetected[i][0] = 'img'

            STDestimationFlag=0
            Alpha =0.5 # Alpha = P(h=1) = prior probability of the Gaussian in the EM formulation
            Temperature=1  # Temperature for confidence weights

            # Distribution of counts for detections and false alarms
            ShowDistributionFlag=0
            if ShowDistributionFlag:
                F=plt.figure(1)
                colors=['b','r']
                Labels=[1,  0]
                Nbad=np.argwhere(m[:,2]==0).size # Number of False detections
                NGood=np.argwhere(m[:,2]==1).size # number of good detections
                for i in range(2):
                    Inds=np.argwhere(m[:,2]==Labels[i]) #  indices of good and false detections (among the 115 available detections)
                    [Values, Counts] = np.unique(m[ Inds, 0], return_counts=True)
                    p=np.asarray([Values, Counts]).T    # P holds the distribution of predicted counts as a P*2 table. Each row is (predicted count value, how many times it occurs in the data)
                    plt.plot(p[:,0],p[:,1],color=colors[i]) # The distributions of flase alarm counts (red) and true positive counts (blue)
                     #    We can see the blue is unimodal, similar to Gaussian, and the red is close to uniform
                plt.show()


            # Here we try to estimate the mean and std of the count for images.
            # images are considered as sets of detected rectangles. We think of each image as describing a different field.
            # The mean and variance we want to estimate are of the spikelet count (per spike) in the field
            # Here field means are estimated from single image (each image is
            # considered a field), but the idea can be trivially extended to estimating mean and var from multiple images of the same field.
            ImageNums=np.unique(m[:,4])  # list of image indices
            ImagesNum=ImageNums.shape[0]

            # initialize all statistics
            MeanEstflag=np.zeros(ImagesNum, dtype=bool)
            MeanEstWithFAFlag=np.zeros(ImagesNum, dtype=bool)
            StdEstflag=np.zeros(ImagesNum, dtype=bool)
            StdEstWithFAFlag=np.zeros(ImagesNum, dtype=bool)
            Ns=np.zeros(ImagesNum)
            TrueM=np.zeros(ImagesNum)  # GT mean
            TrueS=np.zeros(ImagesNum)  # GT std
            EstM=np.zeros(ImagesNum)   # plain estimation of mean
            EstS=np.zeros(ImagesNum)   # plain estimation of std
            EstMO=np.zeros(ImagesNum)  # mean estimation  with oracle
            EstSO=np.zeros(ImagesNum)  # std estimation with oracle
            EstM_EM=np.zeros(ImagesNum)  # Plain EM mean
            EstS_EM=np.zeros(ImagesNum)  # plain EM std
            EstMC=np.zeros(ImagesNum)    # Confidence based mean
            EstSC=np.zeros(ImagesNum)    # confidence based std
            EstMM=np.zeros(ImagesNum)    # median mean
            EstSM=np.zeros(ImagesNum)    # median based std - not used
            EstM_EMC=np.zeros(ImagesNum)   # EM with confidence - mean
            EstS_EMC=np.zeros(ImagesNum)   # EM with confidence - std
            EstMCM=np.zeros(ImagesNum)   # median with confidence weights
            EstSCM=np.zeros(ImagesNum)   # std from  median with confidence weights? - not used
            ImageSpikes=[None]*ImagesNum # Keep the detection sets


            FA_pred = m[np.argwhere(m[:,2]==0),0].squeeze()
            all_pred_range = np.max(m[:,0])-np.min(m[:,0])

            #pred_range=np.max(FA_pred)-np.min(FA_pred)
            FA_mean = 10 #np.mean(FA_pred)
            FA_std = 4 #np.std(FA_pred)
            #print("FA_mean", FA_mean)
            #print("FA_std", FA_std)

            #Compute the estimations, looping over samples
            for i in range(ImagesNum): # collect only 'interesting' images: those for which some spikes are detected, some don't
                inds=np.argwhere(m[:,4]==ImageNums[i]) # find indices of all the detections belonging to image ImageNums(i)
                Spikes= m[inds.squeeze(1),0:4] # get the spikes detections of ImageNums(i)
                N=Spikes.shape[0]
                TrueInds= np.argwhere(Spikes[:,2]==1) # indices of real Spikes

                # Flags indicating which field is relevant to which computation
                MeanEstflag[i] =  TrueInds.shape[0]>0 # Should this field  be used for mean estimation?
                MeanEstWithFAFlag[i]=  not ( TrueInds.shape[0]==N or TrueInds.shape[0]==0)  # Should this field  be used for std estimatuon with FAs?
                StdEstflag[i] =  TrueInds.shape[0]>1 and  np.std(Spikes[TrueInds,1])>0  #  Should this field  be used for std estimatuon?
                                        # only if there are at least two real samples and thery are different (i.e. the real std is >0)
                StdEstWithFAFlag[i]= StdEstflag[i] and not( TrueInds.shape[0]==N or  TrueInds.shape[0]<=1) # Should this field  be used for std estimatuon with FAs?

                # store the image example
                Ns[i]=N # number of spike detections in the image

                # get all gt data
                notFound_inds = np.argwhere(Undetected[:,0] == ImageNums[i])
                notFound = Undetected[notFound_inds.squeeze(1), 2]

                found=Spikes[TrueInds,1]
                trueCounts=[]
                for f in found:
                    trueCounts.append(f[0])
                for u in notFound:
                    trueCounts.append(u)
                trueCounts=np.array(trueCounts)

                if MeanEstflag[i]:
                    # ground truth
                    TrueM[i]=np.mean(trueCounts)#Spikes[TrueInds,1]) # true mean (better be estimated also by adding true undeteced spikes)
                    TrueS[i]=np.std(trueCounts)#Spikes[TrueInds,1]) # true std

                    # simple estimation
                    EstM[i]=np.mean(Spikes[:,0])  # mean of network predictions
                    EstS[i]=np.std(Spikes[:,0])   # std of network predictions

                    # Estimation with a false alarm oracle (not practical - just to provide a bound over the best we can do
                    EstMO[i]=np.mean(Spikes[TrueInds,0]) # mean as estimated by network prediction + an magical oracle of false positives
                    # i,e, what could we have obtained if we had somehow known which detection is a false positive
                    EstSO[i]=np.std(Spikes[TrueInds,0])  # std as estimated by network prediction + a magical oracle of false positives

                    # Robust EM estimator
                    P_UniformD= None #1/all_pred_range  #1/15 #1/pred_range
                    #Preds=Spikes[:,0]# prediction values to estimate the mean from. These are visible values X to explain
                    Preds=[]
                    for s in Spikes:
                       Preds.append(s[0])
                    Preds=np.array(Preds)

                    EstM_EM[i], EstS_EM[i] = EMfunc(Preds,P_UniformD,Alpha, FA_mean, FA_std) #GaussAndUniformEM(Preds,P_UniformD,Alpha) #TwoGaussiansEM

                    # Estimating with Weighting by detector confidence
                    orig_conf=Spikes[:,3]
                    if use_beta:
                        conf= 1/(1+((1-orig_conf)/orig_conf)**beta)
                    else:
                        conf =orig_conf

                    EstMC[i] = np.dot(conf.T ,Preds)/np.sum(conf)
                    EstSC[i] = np.sqrt( np.dot(conf.T, np.square( Preds-EstMC[i])) / np.sum(conf))

                    # median
                    EstMM[i] = np.median(Preds)
                    EstSM[i] = np.nan

                    # EM estimator with confidence priors
                    #GaussProbs = Spikes[:, 3]

                    ###################################################################
                    #GaussProbs = []
                    #for s in Spikes:
                    #    GaussProbs.append(s[3])
                    #GaussProbs = np.array(GaussProbs)
                    ###################################################################
                    GaussProbs = np.array(conf,dtype=float)

                    GaussProbs = 1 / (1 + np.exp(- (np.log(GaussProbs/ (1 - GaussProbs)) ) / Temperature) )  # temperature
                    # if temperature is high, we are less confident in our binary spike / non spike decisions

                    EstM_EMC[i], EstS_EMC[i] = EMfunc(Preds, P_UniformD, GaussProbs, FA_mean, FA_std) #TwoGaussiansEM #GaussAndUniformEM(Preds, P_UniformD, GaussProbs)

                    # median with confidence weights
                    SortedPreds   =np.sort(Preds)
                    arginds =np.argsort(Preds)
                    SortedGaussProbs=GaussProbs[arginds]
                    SecondHalf=np.argwhere( np.cumsum(SortedGaussProbs)/np.sum(SortedGaussProbs) > 0.5)
                    ind=SecondHalf[0]
                    if ind==0:
                        EstMCM[i] = SortedPreds[0]
                    else:
                        EstMCM[i] = ( SortedPreds[ind-1]*SortedGaussProbs[ind-1] +  SortedPreds[ind]*SortedGaussProbs[ind] )/   \
                            (SortedGaussProbs[ind-1] +SortedGaussProbs[ind])
                        #EstMCM[i] = SortedPreds[ind-1];
                    EstSCM =np.nan

                    ImageSpikes[i]=Spikes

                else:
                    a=1

            # Mean estimation - only with false alarms
            MethodKeys=[ 'Standard estimation'  , 'FA oracle estimtion' , 'EM estimation' , 'Confidence estimation' , 'Median' , \
                'EM_with_Confidence' , 'Confidence Median']

            MeasurementKeys=['Mean absolute error' , 'MSE' , 'Relative Error', '1-FUV' ]

            Estimators= np.vstack( [ EstM[MeanEstWithFAFlag] , EstMO[MeanEstWithFAFlag] , EstM_EM[MeanEstWithFAFlag] , \
                                     EstMC[MeanEstWithFAFlag] , EstMM[MeanEstWithFAFlag] , EstM_EMC[MeanEstWithFAFlag], \
                                     EstMCM[MeanEstWithFAFlag]  ])
            Mean_Accuracy_FA= evaluateEstimation( TrueM[MeanEstWithFAFlag],Estimators)

            print('field:{} , min conf:{},  beta:{}'.format(field, min_conf,str(beta)))
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
            #export detections info
            f = open(csv_file, 'a', newline='')
            with f:
                writer = csv.writer(f)
                if line==0:
                    writer.writerow(csv_columns)
                    line=1
                for j in range(len(MethodKeys)):
                    myrow = []
                    myrow.append(field)
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
                q=np.vstack( TrueS[StdEstWithFAFlag] , Estimators )
                np.corrcoef(q[:,2:].T)

