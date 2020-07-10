""" Loads pre-processed data and divides into folds. Also provides a generator that yields data per batches."""

import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

class mydataloader_cv():
    def __init__(self):
        # super().__init__()
        print(" ")

    def load_data(self, path, K=5, max_n_files=None, start_range=None, end_range=None):
        self.path = path
        self.K = K  # number of folds for cross-validation

        # find list of all files in folder
        file_list = [entry.path for entry in os.scandir(path)]

        if max_n_files is not None:  # we decide to load less files (for tests, to be faster)
            if len(file_list) > max_n_files:
                file_list = file_list[:max_n_files]
            else:
                print('There are less files than the provided maximum')

        if start_range is not None and end_range is not None:
            assert end_range >= start_range, "End range has to be greater or equal start range"
            if end_range > len(file_list) - 1: end_range = len(file_list);
            file_list = file_list[start_range - 1:end_range]

        # Here, load individually each patient and then use referencing for creating cross-validation sets

        # load them in a list [n_subjects] each containing an ndarray (patient_epochs, 2, epoch_length)
        self.n_subjects = len(file_list)
        self.pEOG = [[] for _ in range(self.n_subjects)]
        self.hypnogram = [[] for _ in range(self.n_subjects)]
        self.PSD = [[] for _ in range(self.n_subjects)]
        self.PSD_highres = [[] for _ in range(self.n_subjects)]
        self.spect = [[] for _ in range(self.n_subjects)]
        self.stats = [[] for _ in range(self.n_subjects)]
        self.AR_coefs = [[] for _ in range(self.n_subjects)]

        # these two variables will contain ALL the data. Then, it is needed to create the cv version, which references this one
        for i, fullpath in enumerate(file_list):
            if os.path.isfile(fullpath):
                # load the data
                with open(fullpath, 'rb') as filehandle:
                    data_dict = pickle.load(filehandle)
                print(fullpath[-17:-5] + ' loaded!')

                # recover variables
                self.pEOG[i] = data_dict['EOG']
                self.hypnogram[i] = data_dict['Hypnogram']
                self.PSD[i] = data_dict['PSD']
                self.PSD_highres[i] = data_dict['PSD high resolution']
                self.spect[i] = data_dict['Spectrogram']
                self.stats[i] = data_dict['Statistics']
                self.AR_coefs[i] = data_dict['AR coefficients']

        self.pEOG = np.asarray(self.pEOG)
        self.hypnogram = np.asarray(self.hypnogram)
        self.PSD = np.asarray(self.PSD)
        self.PSD_highres = np.asarray(self.PSD_highres)
        self.spect = np.asarray(self.spect)
        self.stats = np.asarray(self.stats)
        self.AR_coefs = np.asarray(self.AR_coefs)

        kfold = KFold(n_splits=self.K, shuffle=True, random_state=42)
        kfold.get_n_splits(file_list)
        self.validation_subjects_cv = [[] for _ in range(self.K)]
        self.pEOG_cv_train = [[] for _ in range(self.K)]
        self.pEOG_cv_valid = [[] for _ in range(self.K)]
        self.hypnogram_cv_train = [[] for _ in range(self.K)]
        self.hypnogram_cv_valid = [[] for _ in range(self.K)]
        self.PSD_cv_train = [[] for _ in range(self.K)]
        self.PSD_cv_valid = [[] for _ in range(self.K)]
        self.PSD_highres_cv_train = [[] for _ in range(self.K)]
        self.PSD_highres_cv_valid = [[] for _ in range(self.K)]
        self.spect_cv_train = [[] for _ in range(self.K)]
        self.spect_cv_valid = [[] for _ in range(self.K)]
        self.stats_cv_train = [[] for _ in range(self.K)]
        self.stats_cv_valid = [[] for _ in range(self.K)]
        self.AR_coefs_cv_train = [[] for _ in range(self.K)]
        self.AR_coefs_cv_valid = [[] for _ in range(self.K)]

        i = 0
        for train_index, val_index in kfold.split(file_list):
            print('Validation subjects in fold ' + str(i),
                  val_index)  # just to check that it is always the same, and that we can 'join' NN of the same fold
            self.validation_subjects_cv[i] = val_index
            self.pEOG_cv_train[i] = np.concatenate((self.pEOG[train_index]), axis=0)
            self.pEOG_cv_valid[i] = np.concatenate((self.pEOG[val_index]), axis=0)
            self.hypnogram_cv_train[i] = np.concatenate((self.hypnogram[train_index]))
            self.hypnogram_cv_valid[i] = np.concatenate((self.hypnogram[val_index]))
            self.PSD_cv_train[i] = np.concatenate((self.PSD[train_index]), axis=0)
            self.PSD_cv_valid[i] = np.concatenate((self.PSD[val_index]), axis=0)
            self.PSD_highres_cv_train[i] = np.concatenate((self.PSD_highres[train_index]), axis=0)
            self.PSD_highres_cv_valid[i] = np.concatenate((self.PSD_highres[val_index]), axis=0)
            self.spect_cv_train[i] = np.concatenate((self.spect[train_index]), axis=0)
            self.spect_cv_valid[i] = np.concatenate((self.spect[val_index]), axis=0)
            self.stats_cv_train[i] = np.concatenate((self.stats[train_index]), axis=0)
            self.stats_cv_valid[i] = np.concatenate((self.stats[val_index]), axis=0)
            self.AR_coefs_cv_train[i] = np.concatenate((self.AR_coefs[train_index]), axis=0)
            self.AR_coefs_cv_valid[i] = np.concatenate((self.AR_coefs[val_index]), axis=0)

            i += 1

    # redefines the batch generator so that it takes the corresponding training or validation set
    def batch_generator(self, kfold, batch_size=512, validation_set=False, shuffle=True, shuffling_seed=42):
        # returns a generator object which can be iterated using next() and yields EOG (batch_size x 2 x 1500)
        # (corresponding to batch size, left+right, 30 seconds * Fs)
        # and hypnogram (batch_size), our labels for the supervised problem
        self.current_kfold = kfold
        self.batch_size = batch_size

        if validation_set is False:
            pEOG = self.pEOG_cv_train[kfold]
            hypnogram = self.hypnogram_cv_train[kfold]
            PSD = self.PSD_cv_train[kfold]
            PSD_highres = self.PSD_highres_cv_train[kfold]
            spect = self.spect_cv_train[kfold]
            stats = self.stats_cv_train[kfold]
            AR_coefs = self.AR_coefs_cv_train[kfold]
        elif validation_set is True:
            pEOG = self.pEOG_cv_valid[kfold]
            hypnogram = self.hypnogram_cv_valid[kfold]
            PSD = self.PSD_cv_valid[kfold]
            PSD_highres = self.PSD_highres_cv_valid[kfold]
            spect = self.spect_cv_valid[kfold]
            stats = self.stats_cv_valid[kfold]
            AR_coefs = self.AR_coefs_cv_valid[kfold]

        else:
            print(
                'Please, input a valid value (True or False) for "validation_set" parameter, which is to be False for '
                'generation of training batches and true for generation of validation batches')

        # total number of batches we can make
        n_batches = pEOG.shape[0] // batch_size
        if validation_set is False:
            self.n_batches = n_batches  # declare this so it can be inspected from the outside
        elif validation_set is True:
            self.n_batches_valid = n_batches

        # Keep only enough sleep epochs to make full batches
        pEOG = pEOG[:n_batches * batch_size, :, :]
        hypnogram = hypnogram[:n_batches * batch_size]
        PSD = PSD[:n_batches * batch_size, :, :]
        PSD_highres = PSD_highres[:n_batches * batch_size, :, :]
        spect = spect[:n_batches * batch_size, :, :, :]
        stats = stats[:n_batches * batch_size, :, :]
        AR_coefs = AR_coefs[:n_batches * batch_size, :, :]

        # reshape to (batch_size, n_batches, __remaining dimensions__)
        pEOG = pEOG.reshape(batch_size, n_batches, 2, 1500)  # 2 (left and right EOG), epoch_length=1500
        hypnogram = hypnogram.reshape(batch_size, n_batches)
        PSD = PSD.reshape(batch_size, n_batches, 2, 26)  # freq resolution 1 Hz
        PSD_highres = PSD_highres.reshape(batch_size, n_batches, 2, 101)  # freq resolution 0.25 Hz
        spect = spect.reshape(batch_size, n_batches, 2, 26, 59)  # 26 --> freq bins; 59 --> 50% overlapping 1s time bins
        stats = stats.reshape(batch_size, n_batches, 2, 6)  # 6 stats: min, max, mean, var, skew, kurt
        AR_coefs = AR_coefs.reshape(batch_size, n_batches, 2, 9)

        if shuffle:  # swap samples to avoid training in a single direction in each batch
            np.random.seed(shuffling_seed)  # use seed to have repeatable results

            # swap dims 0 and 1 = batch_size and n_batches # joined in v4
            mask0 = np.random.permutation(np.arange(pEOG.shape[0])).reshape(-1, 1)
            mask1 = np.random.permutation(np.arange(pEOG.shape[1]))
            pEOG = pEOG[mask0, mask1, :, :]
            # later we will select in dim 1 (so we end up with batch_size x 2 x 1500); use squeeze!
            hypnogram = hypnogram[mask0, mask1]
            PSD = PSD[mask0, mask1, :, :]
            PSD_highres = PSD_highres[mask0, mask1, :, :]
            spect = spect[mask0, mask1, :, :, :]
            stats = stats[mask0, mask1, :, :]
            AR_coefs = AR_coefs[mask0, mask1, :, :]

            # make a detection to generate again the batches and a new iterator when current_batch = n_batches-1
        for i in range(n_batches):
            self.current_batch = i
            eog = pEOG[:, i, :, :]
            h = hypnogram[:, i]
            psd_coeffs = PSD[:, i, :, :]
            mse_coeffs = np.zeros((PSD.shape))
            psd_highres_coeffs = PSD_highres[:, i, :, :]
            spect_values = spect[:, i, :, :, :]
            stats_values = stats[:, i, :, :]
            AR_coefs_values = AR_coefs[:, i, :, :]
            yield (eog, h, mse_coeffs, psd_coeffs, psd_highres_coeffs, spect_values, stats_values, AR_coefs_values)
