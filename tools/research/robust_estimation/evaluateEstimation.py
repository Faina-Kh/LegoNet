# Estimate the accuracy of several  regression estimators w.r.t the ground truth gt
# Input:
#   gt - a 1D vector of true values to regress. size N
#   Estimators - K*N matrix containing 1 or more estimators of gt
# Output:
#   Table - a K*4 accuracy evaluation matrix results
#       column 1 - the mean absolute difference of the K estimators
#       column 2 - the mean square error of the K estimators
#       column 3 - the relative error of the K estimators (mean of absolute error in fraction of the true value )
#       column 4 - the (1-Fraction of unexplained variance) of the K estimators

import numpy as np

def evaluateEstimation(gt, Estimators):

    Table=np.zeros([Estimators.shape[0],4])

    for i in range(Estimators.shape[0]):
        Estimator=Estimators[i,:]
        EstL1 = np.mean(np.abs(gt-Estimator))   # abs count difference
        EstMSE = np.mean((gt-Estimator)**2)   # MSE
        RelError= np.mean(np.abs(gt-Estimator)/gt)  # Relative error
        ExplainedVar= 1- EstMSE/np.var(gt, ddof=1) # 1-Fraction of variance unexplained

        Table[i,0]=EstL1
        Table[i,1]=EstMSE
        Table[i,2]=RelError
        Table[i,3]=ExplainedVar

    return Table

