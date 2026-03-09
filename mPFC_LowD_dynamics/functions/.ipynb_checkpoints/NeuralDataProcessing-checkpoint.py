import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# from sklearn.decomposition import FactorAnalysis
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# from astropy import convolve
import pdb
import numpy as np

def get_condition_average_activity(myData, TrialTags, TimePoints = None):
    """Calculate the average conditioned neuron activity.

    Args:
        myData (object): The data object containing the spike train data.
        TrialTags (list): List of trial tags for conditions of interests.
        TimePoints (bool array): Boolean array indicating the time points of interest.

    Returns:
        ndarray: The average conditioned neuron activity array.

    """
    if TimePoints is None:
        TimePoints = np.ones_like(myData.timevec).astype(bool)
        T = len(myData.timevec[TimePoints])
    else:
        T = TimePoints.sum()  # Number of time points to process
    C = len(TrialTags)  # Number of conditions
    N = myData.NeuralTableByTrial.shape[1]  # Number of neurons
    average_conditioned_activity = np.zeros((T, C, N)) # Initialize the neural data matrix

    for tagindex, tag in enumerate(TrialTags):
        AverageActivity = myData.AverageAcrossTrials(TrialsOfInterest=myData.TrialsOfInterest[tag])
        for unitIndex, unit in enumerate(np.arange(N)):
            average_conditioned_activity[:, tagindex, unitIndex] = \
                AverageActivity['SpikeTrainUnit_' + str(unit)][TimePoints]

    return average_conditioned_activity

def get_single_trial_activity(myData, TimePoints = None):
    if TimePoints is None:
        TimePoints = np.ones_like(myData.timevec).astype(bool)
        T = len(myData.timevec[TimePoints])
    else:
        T = TimePoints.sum()  # Number of time points to process
    C = myData.NeuralTableByTrial.shape[0]  # Number of trials
    N = myData.NeuralTableByTrial.shape[1]  # Number of neurons
    single_trial_activity = np.zeros((T, C, N)) # Initialize the neural data matrix

    for trial in np.arange(C):
        for unitIndex, unit in enumerate(np.arange(N)):
            single_trial_activity[:, trial, unitIndex] = \
                myData.NeuralTableByTrial['SpikeTrainUnit_' + str(unit)].iloc[trial][TimePoints]

    return single_trial_activity

def normalized_gaussian_kernel(kernel_std = 3):
    """ make a gaussian filter with std of 3 timepoints (i.e. std = 3/Fs)
        kernel size should usually be 6 * kernel_std so as to be +-3 std"""
    kernel_size = np.round(kernel_std*6)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    kernel = signal.windows.gaussian(kernel_size, std=kernel_std) # First input is the size of the smoothing window
    kernel = kernel/np.sum(kernel) # Normalize the kernel    
    return kernel

def convolve_with_filter(activity, filt, mode):
    """Convolve the spike histograms with a Gaussian filter.

    Args:
        activity (ndarray): Neural activity array.
        filt (ndarray): The Gaussian kernel to convolve the spike histograms.

    Returns:
        ndarray: The convolved neural activity array.

    """
    convolved_activity = np.apply_along_axis(
        lambda m: np.convolve(m, filt, mode=mode),
        axis=0, arr=activity)

    return convolved_activity

def apply_pca(pca_input,n_components = 8, plot = False):
    """ applying PCA to Neural data

    Args: 
        pca_input (ndarray) : (Time x Condition, Neurons) - here neurons are the dimension to reduce
        n_components (int) : number of PCs

    Returns:
        PCA_model
    """
    
    PCA_model = PCA(n_components)
    PCA_model.fit(pca_input)
    pca_score = PCA_model.explained_variance_ratio_
    
    # Plot variance explained by PCs.
    if plot:
        plt.figure(figsize=(5, 3))
        plt.stem(np.arange(1,pca_score.shape[0]+1,1),pca_score)
        plt.title('Variance Explained by each PC')
        plt.xlabel('Principle components')
        plt.ylabel('Variance Explained')
        plt.show()

    return PCA_model

###### TODO: Add other analytical methods (e.g., dPCA, GLMs) to this file.

    