
# Compute the parameters if a Gaussian in a mixture of A gaussian and a uniform distribution (for 1d scalar values )
# Input:
#   Data - a column vactor of values (the sample)
#   P_UniformD - a scalar. The density of the uniform
#   P_His1 - two options:
#            If this is a scalar, it is the prior probability P(h=1) i.e. the probability to belong to the Guassian
#            If this is a 1D vector of size like Data, each point has a different prior P(h_i=1) stated by this vector
# Output:
#   m - the mean of the estimated Gaussian
#   s = the std of the estimated Gaussian
#
# We assume there is a hidden variable h for each x, stating if it is from the Gaussian (a true spikelet count)
# or from the uniform (a false count). h=1 means x is from the Gaussian, h=0 mean it is from the uniform


import numpy as np

def GaussAndUniformEM(Data, P_UniformD, P_His1, FA_mean, FA_std):
    if type(P_His1)==float or P_His1.shape==():  # if P_His1 is a scalar
        P_His1=P_His1*np.ones(Data.shape[0])

    GaussMean = np.mean(Data) # initial estimation of mean
    GaussSig = np.std(Data) # initial estimation of std


    if GaussSig>0:  # Do EM if the predictions are not all the same
        for k in range(1000):   #EM - run several iterationss
            prevM = GaussMean

            # E -step
            P_XGaussian = (1/(2*np.pi*GaussSig)) * np.exp(-0.5 * ((Data- GaussMean)**2)/GaussSig**2 ) # P(X|h=1)  is Gaussian
            P_HisGauss_GivenX =  (P_His1 * P_XGaussian)/(P_His1*P_XGaussian + (1-P_His1)*P_UniformD)  # P(h=1|X) using Bayes
            # M -step
            GaussMean = np.dot(P_HisGauss_GivenX.T,Data) / np.sum(P_HisGauss_GivenX) # Gauss mean estimated based on h probability (p(h=1|x) are the weights)
            GaussSig = np.sqrt( np.dot(P_HisGauss_GivenX.T, (Data - GaussMean)**2)  / np.sum(P_HisGauss_GivenX) )  #Gauss STD estimated from h probability
            if GaussSig<0.01: # if we approach very small std, break to avoid numerical stability problems
                break

            if k>0 and (np.abs(prevM-GaussMean)/prevM)<0.001:
                print(k)
                break
    else:
        GaussSig=0  # Do not run EM if the sigma is 0 to begin with

    return GaussMean, GaussSig


def TwoGaussiansEM(Data, P_UniformD, P_His1,  FA_mean, FA_std, relevant_out=None):
    if type(P_His1)==float or P_His1.shape==():  # if P_His1 is a scalar
        P_His1=P_His1*np.ones(Data.shape[0])

    GaussMean = np.mean(Data) # initial estimation of mean
    GaussSig = np.std(Data) # initial estimation of std

    f_m = FA_mean #16.02 #30 #43 #
    f_s = FA_std #4.27 #10 #17#

    if GaussSig>0:  # Do EM if the predictions are not all the same
        for k in range(1000):

            prevM_1 = GaussMean
            prevM_2 = f_m

            #EM - run several iterationss
            # E -step
            P_XGaussian = (1/(2*np.pi*GaussSig)) * np.exp(-0.5 * ((Data- GaussMean)**2)/GaussSig**2 ) # P(X|h=1)  is Gaussian
            P_XGaussian_2 = (1/(2 * np.pi * f_s)) * np.exp( -0.5 * ((Data - f_m) ** 2) / f_s ** 2)  # P(X|h=0)  is Gaussian

            P_HisGauss_GivenX =  (P_His1 * P_XGaussian)/(P_His1*P_XGaussian + (1-P_His1)*P_XGaussian_2)  # P(h=1|X) using Bayes
            P_HisGauss_GivenX_2 = ((1 - P_His1) * P_XGaussian_2) / (P_His1 * P_XGaussian + (1 - P_His1) * P_XGaussian_2)  # P(h=0|X) using Bayes

            # M -step
            GaussMean = np.dot(P_HisGauss_GivenX.T,Data) / np.sum(P_HisGauss_GivenX) # Gauss mean estimated based on h probability (p(h=1|x) are the weights)
            GaussSig = np.sqrt( np.dot(P_HisGauss_GivenX.T, (Data - GaussMean)**2)  / np.sum(P_HisGauss_GivenX) )  #Gauss STD estimated from h probability

            f_m = np.dot(P_HisGauss_GivenX_2.T, Data) / np.sum(P_HisGauss_GivenX_2)  # Gauss mean estimated based on h probability (p(h=0|x) are the weights)
            f_s = np.sqrt(np.dot(P_HisGauss_GivenX_2.T, (Data - f_m) ** 2) / np.sum(P_HisGauss_GivenX_2))  # Gauss STD estimated from h probability

            if GaussSig<0.01 or f_s <0.01: # if we approach very small std, break to avoid numerical stability problems
                break

            # check if converged
            if (np.abs(prevM_1-GaussMean)/prevM_1)<0.001 and (np.abs(prevM_2-f_m)/prevM_2)<0.001:
                #print(k)
                break

    else:
        GaussSig=0  # Do not run EM if the sigma is 0 to begin with
        return GaussMean, GaussSig


    if relevant_out == "sum":
        GaussSum = np.dot(P_HisGauss_GivenX.T,Data)
        return GaussSum, GaussSig # sum of network predictions
    elif relevant_out == "avg":
        return GaussMean, GaussSig  # mean of network predictions



    # if GaussMean>f_m:
    #     return GaussMean, GaussSig
    # else:
    #     return f_m, f_s