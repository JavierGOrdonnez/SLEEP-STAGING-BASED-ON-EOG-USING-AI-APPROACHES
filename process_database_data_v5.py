# v5: Included AR coefficients
# as additional features that may be useful for classification

# IMPORTANT: MSE takes a huge amount of time and I have found it is not that relevant
# therefore it wont be computed and included

import os
import pickle
import numpy as np
import pyedflib
from xml.dom import minidom
from scipy import signal
from scipy.signal import butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg
import scipy


def process_database_data(PSG_filepath, hypnogram_filepath, target_folderpath, remove_noise=True, datatype="float32",
                          force_reprocessing=False):
    # check target directory, if this file already exists
    if not os.path.isdir(target_folderpath): os.mkdir(target_folderpath)
    if target_folderpath[-1] is not '/': target_folderpath = target_folderpath + '/'
    name = PSG_filepath[-16:-4]
    file_path = target_folderpath + name + '.data'
    if os.path.isfile(file_path) and force_reprocessing is False:
        # if the file already exists in the target folder and we are not purposefully forcing to recompute all files, lets exit the function
        print('File '+file_path+' already exists. Skipped.')
        return 0
    # else, let's continue with it and execute processing

    f = pyedflib.EdfReader(PSG_filepath)
    signal_labels = f.getSignalLabels()

    # gets and saves left EOG signal
    idx = next((idx for idx, value in enumerate(signal_labels) if value == 'EOG(L)'))
    EOG_left = f.readSignal(idx)
    idx = next((idx for idx, value in enumerate(signal_labels) if value == 'EOG(R)'))
    EOG_right = f.readSignal(idx)
    f.close()

    xmldoc = minidom.parse(hypnogram_filepath)
    SleepStages = xmldoc.getElementsByTagName('SleepStage')
    hypnogram = np.asarray([int(item.childNodes[0]._data) for item in SleepStages])

    # Merge sleep stages S3 and S4 (R&K) into N3 (AASM)
    hypnogram[hypnogram == 4] = 3
    # Put REM class as 4 to avoid the CUDA error: device-side assert triggered
    hypnogram[hypnogram == 5] = 4

    pEOGleft = []  # p stands for partitioned
    pEOGright = []
    n = len(hypnogram)
    Fs = 50  # we know this from database documentation
    # we also know that each PSG interval is 30 seconds, that is standard
    assert n * 30 * Fs == len(EOG_left), "The length of the hypnogram and the signal do not match"
    for i in range(n):
        pEOGleft.append(EOG_left[i * Fs * 30:(i + 1) * Fs * 30])
        pEOGright.append(EOG_right[i * Fs * 30:(i + 1) * Fs * 30])

    # normalize EOG amplitude
    pEOGleft = pEOGleft / np.max(np.abs(pEOGleft))
    pEOGright = pEOGright / np.max(np.abs(pEOGright))

    if remove_noise:  # find variance of each segment and eliminate those that are electronic noise (v > 0.75)
        rEOGleft = []
        rEOGright = []
        rhypnogram = []  # r stands for refined
        eliminated = []
        for i in range(len(pEOGleft)):
            if (np.var(pEOGleft[i]) < 0.75  # ensure variance is not to high (detached electrodes, just noise)
                    and hypnogram[i] >= 0 and hypnogram[i] <= 5):  # ensure proper values of hypnogram (elim unknowns)
                rEOGleft.append(pEOGleft[i])
                rEOGright.append(pEOGright[i])
                rhypnogram.append(hypnogram[i])
            else:
                eliminated.append(i)
        print('Eliminated segments (of %d): ' % ((len(pEOGleft) - 1)), eliminated)

        pEOGleft = rEOGleft
        pEOGright = rEOGright
        hypnogram = rhypnogram

    pEOGleft = np.asarray(pEOGleft, dtype=datatype)
    pEOGright = np.asarray(pEOGright, dtype=datatype)
    pEOG = np.stack((pEOGleft, pEOGright), axis=2)
    pEOG = np.swapaxes(pEOG, 1, 2)
    hypnogram = np.asarray(hypnogram, dtype="int")

    # with clean data, let's create additional measures, and store them:

    # IMPORTANT: MSE takes a huge amount of time and I have found it is not that relevant
    # therefore it wont be computed and included
    # MSE
    # fulldata_std = np.std(pEOG)
    # include only left EOG? compare both (check correlation, relative difference...)
    # correlation of 0.895 --> basically the same
    # mean relative diff per scale are very low (max 3%)
    # but mean diff per sample can be much bigger (up to >30%)
    # mse_coeffs = np.array([[MSE(pEOG[i, 0, :], min_scale=1, max_scale=20, m=2, r=0.2 * fulldata_std),
    #                         MSE(pEOG[i, 1, :], min_scale=1, max_scale=20, m=2, r=0.2 * fulldata_std)]
    #                        for i in range(len(pEOG))], dtype=datatype)

    # PSD (Welch method)
    psd_coeffs = np.array([[signal.welch(pEOG[i, 0, :], fs=Fs, window='hamming', nperseg=Fs, scaling='spectrum')[1],
                            signal.welch(pEOG[i, 1, :], fs=Fs, window='hamming', nperseg=Fs, scaling='spectrum')[1]]
                           for i in range(len(pEOG))], dtype=datatype)

    # PSD (Welch method) ; higher resolution
    psd_highres = np.array(
        [[signal.welch(pEOG[i, 0, :], fs=Fs, window='hamming', nperseg=Fs * 4, scaling='spectrum')[1],
          signal.welch(pEOG[i, 1, :], fs=Fs, window='hamming', nperseg=Fs * 4, scaling='spectrum')[1]]
         for i in range(len(pEOG))], dtype=datatype)

    # spectrogram
    # using tukey window 25%, as Zhang 2019
    spect = np.array([[signal.spectrogram(pEOG[i, 0, :], fs=Fs, nperseg=2*Fs, noverlap=int(Fs/2),
                                          scaling='spectrum')[2],
                       signal.spectrogram(pEOG[i, 1, :], fs=Fs, nperseg=2*Fs, noverlap=int(Fs/2),
                                          scaling='spectrum')[2]]
                      for i in range(len(pEOG))], dtype=datatype)

    # useful statistics
    stats = np.array([[[np.min(pEOG[i, 0, :]), np.max(pEOG[i, 0, :]), np.mean(pEOG[i, 0, :]), np.var(pEOG[i, 0, :]),
                        scipy.stats.skew(pEOG[i, 0, :]), scipy.stats.kurtosis(pEOG[i, 0, :])],
                       [np.min(pEOG[i, 1, :]), np.max(pEOG[i, 1, :]), np.mean(pEOG[i, 1, :]), np.var(pEOG[i, 1, :]),
                        scipy.stats.skew(pEOG[i, 1, :]), scipy.stats.kurtosis(pEOG[i, 1, :])]]
                      for i in range(len(pEOG))], dtype=datatype)

    # AR coefficients up to order 8
    leftEOG = pEOG[:, 0, :].reshape(-1)
    rightEOG = pEOG[:, 1, :].reshape(-1)
    b, a = butter(N=8, Wn=[4, 8], btype='bandpass', fs=Fs)
    filtered_leftEOG = filtfilt(b, a, leftEOG)
    filtered_rightEOG = filtfilt(b, a, rightEOG)

    AR_coefs = np.zeros((pEOG.shape[0], 2, 9))  # order 8 returns 9 parameters

    pleftEOG = filtered_leftEOG.reshape(pEOG.shape[0], 1500)
    for j, part in enumerate(pleftEOG):
        model = AutoReg(part, lags=8).fit()
        AR_coefs[j, 0, :] = model.params
    prightEOG = filtered_rightEOG.reshape(pEOG.shape[0], 1500)
    for j, part in enumerate(prightEOG):
        model = AutoReg(part, lags=8).fit()
        AR_coefs[j, 1, :] = model.params

    # now let's save into a file
    data_dict = {'Patient_code': name, 'EOG_Fs': Fs, 'Hypnogram': hypnogram, 'EOG': pEOG,  # 'MSE': mse_coeffs,
                 'PSD': psd_coeffs, 'PSD high resolution': psd_highres,
                 'Spectrogram': spect, 'Statistics': stats, 'AR coefficients': AR_coefs}
    with open(file_path, 'wb') as filehandle:
        pickle.dump(data_dict, filehandle)

    return 1
