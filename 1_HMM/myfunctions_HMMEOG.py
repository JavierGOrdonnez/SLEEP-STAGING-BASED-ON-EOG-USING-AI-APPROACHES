import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix as cm
from scipy import signal
from scipy.fftpack import fft, fftshift



def hypnogram_vec2mat(hypnogram, clusters): 
    '''
    Inputs:
        Hypnogram (original): a list of elements [0,0,0,0,....,3,3,3,....,0,0,0]
        Clusters: a list of lists, with the packages that are desired
                for example, [ [0], [1,2,3,4], [5] ] means that sleep stages 1-4 must be
                gathered together into a single class
    Outputs:
        Hypnogram_matrix, ndarray of dimensions (len(hypnogram),len(clusters))
    '''

    # let's create the direct mapping
    in2out = np.ndarray((6, 1), dtype=int)
    for i in range(0, 5 + 1):  # this is always, it is the format of PSG
        for n, elem in enumerate(clusters):
            if i in elem: in2out[i] = n; break
    #     print(in2out)

    hypnogram_matrix = np.zeros((len(hypnogram), len(clusters)))
    for i in range(len(hypnogram)):
        hypnogram_matrix[i, in2out[int(hypnogram[i])]] = 1

    return hypnogram_matrix


def hypnogram_mat2vec(hypnogram_matrix):
    hyp_vect = hypnogram_matrix @ np.arange(hypnogram_matrix.shape[1])
    return hyp_vect

def flatten_data(pEOG, H):
    N = pEOG.shape[0]
    L = pEOG.shape[2]
    fullEOG = np.zeros((N*L, 2))  # number of chuncks * length of each chunck
    fullH = np.zeros((N*L, H.shape[1]))
    for i in np.arange(N):
        fullEOG[i * L:(i + 1) * L, :] = pEOG[i,:,:].reshape(L,2)
        fullH[i * L:(i + 1) * L, :] = H[i, :].reshape(1, -1)
    return fullEOG, fullH

def obtain_correlation_matrix(H, Z, makeplots=True, hypnogram_labels=['Wake', 'NREM', 'REM'],mode='sqrt'):
    '''
    Computes a correlation matrix that measures how much an HMM state corresponds with each sleep stage.
    This is done by adding up the probabilities of choosing the state at each time point over which the times at which a given sleep stage was happening.
    That is, CORR(i,j) is the addition of the prob of HMM choosing state I over all times that we were having sleep J
    '''
    corr_matrix = np.zeros((Z.shape[1], H.shape[1])) # N HMM states x M sleep stages
    margZ = np.sum(Z, axis=0)
    margH = np.sum(H, axis=0)  # already compute these marginals, so no need to compute them every time
    for s in range(Z.shape[1]): # for each HMM states
        for n in range(H.shape[1]):
            if mode is 'sqrt': corr_matrix[s, n] = np.dot(H[:, n], Z[:, s]) / np.sqrt(margH[n] * margZ[s])  # normalizing by freq of HMM state and freq of sleep stage
            if mode is 'normal': corr_matrix[s, n] = 10000 * np.dot(H[:, n], Z[:, s]) / (margH[n] * margZ[s])
            if mode is 'nonorm': corr_matrix[s, n] = np.dot(H[:, n], Z[:, s])
    # use 'nonorm', we are normalizing in the plot
    if makeplots: plotCM(corr_matrix)
    return corr_matrix

def plotCM(corr_matrix,labels=None,states=None):
    if labels is None:
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    if states is None:
        states = ['State '+str(i+1) for i in range(corr_matrix.shape[0])]
    else:
        states = ['State '+str(s+1) for s in states]    
    cmap = 'gist_gray'
    cm = corr_matrix.T
    counts_per_class = np.sum(cm,axis=1)
    margZ = np.sum(cm, axis=0)
    counts_per_class[counts_per_class == 0] = 1  # to avoid division by zero
    normcm = (cm.T / counts_per_class).T  # normalize by real classes

    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8,6),gridspec_kw={'width_ratios': [1, 5], 'height_ratios': [5,1]})
    norm_counts_per_class = (counts_per_class / np.sum(counts_per_class)).reshape(-1, 1)
    # relative freq of sleep stages
    sb.heatmap(norm_counts_per_class, ax=ax1, annot=True, fmt='.2f', cmap=cmap, xticklabels=[],
               cbar=False, vmin=0, vmax=0.5)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_yticklabels(labels, rotation=90, fontsize="10", va="center")
    ax1.set_title('Relative frequency of\neach sleep stage')
    
    # correlation between sleep stages and HMM hidden states
#     sb.heatmap(normcm, ax=ax2, annot=True, fmt='.3f', cmap=cmap, linewidths=0.05, cbar=False,
#                 vmin=0, vmax=1, yticklabels=labels)
    sb.heatmap(normcm, ax=ax2, annot=True, fmt='.3f', cmap=cmap, linewidths=0.05, cbar=False,
                vmin=0, vmax=0.6, yticklabels=[], xticklabels=[])
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.set_title('Distribution of sleep stages \namong HMM hidden states')
    
    ax3.axis('off')

    # relative freq of HMM states
    normZ = (margZ / np.sum(margZ) ).reshape(1,4)
    sb.heatmap(normZ, ax=ax4, annot=True, fmt='.2f', cmap=cmap, yticklabels=[],
               xticklabels=states, cbar=False, vmin=0, vmax=0.5)
    ax4.set_xlabel('HMM hidden states relative frequency',fontsize=12)
    ax4.tick_params(axis='both', which='both', length=0)

    plt.show()

def rearrange_states(matrix, state_order): # give manually the state order
    new_matrix = np.zeros((matrix.shape))
    for i in range(matrix.shape[0]):
        new_matrix[i,:] = matrix[state_order[i]]

    return new_matrix # return the rearranged corr matrix and transition matrix 

def rearrange_transmat(matrix, state_order, makeplots=True): # give manually the state order
    new_matrix = np.zeros((matrix.shape))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i,j] = matrix[state_order[i], state_order[j]]
            
    if makeplots:
        states = ['State '+str(s+1) for s in state_order]   
        cmap = 'gist_gray'
        fig, ax1 = plt.subplots(1,1,figsize=(5,5))
        res = sb.heatmap(new_transmat, ax=ax1, annot=True, fmt='.2f', cmap=cmap,
                       xticklabels=states, cbar=False, vmin=0, vmax=0.4)
        for _, spine in res.spines.items():
            spine.set_visible(True)
        plt.title('Transition probabilities between HMM states')
        ax1.tick_params(axis='both', which='both', length=0)
        ax1.set_yticklabels(states, rotation=90, fontsize="10", va="center")
        plt.show()

    return new_matrix # return the rearranged corr matrix and transition matrix 

