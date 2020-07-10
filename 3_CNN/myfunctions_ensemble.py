"""Join several feature extractors (CNN) and combine their outputs through a classifier linear layer"""

import numpy as np
import torch
from torch import nn
import copy
from myfunctions_base_EOGnn_cv import base_EOGnn_cv
from myfunctions_CNN import EOG_1dcnn_cv
from myfunctions_doubleCNN import EOG_double1dcnn_cv


class ensemble(base_EOGnn_cv):

    def __init__(self, list_classifiers, list_epochs, K, lr=0.0001,
                 use_GPU=True, output_classes=5, class_values=None, class_weights='balanced', labels=None,
                 dropout=True, dropout_prob=0.2, optimizer='Adam',
                 savepath='D:/TFM/trained_models/'):
        """
        :param list_classifiers: a list of initialized classifiers, including as attribute (when instantiating them)
         the savepath from which they can be loaded
        :param list_epochs: which epoch to load for each classifier (and all CV folds)

        All other parameters are to be passed to "base_EOGnn_cv", please check their specifications in that class definition.
        """

        super().__init__(K=K, dropout=dropout, dropout_prob=dropout_prob, lr=lr, output_classes=output_classes,
                         class_values=class_values, class_weights=class_weights, labels=labels,
                         use_GPU=use_GPU, savepath=savepath)

        self.type = 'ensemble'

        assert len(list_classifiers) == len(list_epochs), "Input lists should all have the same length"
        self.N = len(list_classifiers)  # how many classifiers we have
        self.classifiers = list_classifiers
        self.epochs = list_epochs
        # self.fixed_parameters = fixed_parameters
        # this parameter is always True (never re-train underlying feature extractors)

        outputsize = 0
        for i in range(self.N):
            outputsize += self.classifiers[i].outputsize
        self.outputsize = outputsize  # to know the input size of the linear layer
        self.classifier_layer = nn.Linear(int(self.outputsize), self.output_classes)
        # logistic classifier, combining extracted features

        self.loaded_fold = None
        self.to(self.device)

        self.create_optimizer(optimizer, params=self.classifier_layer.parameters())

        self.save_initial_state()

    def forward(self, data):
        # only at the start of each fold, load the underlying feature extractors
        if self.loaded_fold != self.current_fold:
            self.load_classifiers(self.current_fold)
            print('Classifiers loaded (in forward)')

        # data is a bunch of different features in dataloader_v5
        # data = (eog, h, mse_coeffs, psd_coeffs, psd_highres_coeffs, spect_values, stats_values, AR_coefs_values)
        features = np.zeros((data[0].shape[0], self.outputsize))  # shape 0 is batch size. All features of same sample are being flattened.
        features = torch.from_numpy(features).to(self.device)
        current_pos = 0

        with torch.no_grad():
            for i in range(self.N):  # get and store the output of each of the included models

                # NOW EOG CNN forward takes in "data" and then does x = data[0]
                # ALL FORWARD METHODS SHOULD TAKE "DATA" AS INPUTS
                output_features, _ = self.loaded_clfs[i].forward(data)
                features[:, current_pos:current_pos + self.loaded_clfs[i].outputsize] = output_features.view(data[0].shape[0], -1)
                current_pos += self.loaded_clfs[i].outputsize

        if self.dropout_bool: features = self.dropout(features)

        # apply a linear layer to these outputs to get a classification
        classes = self.logsoftmax(self.classifier_layer(features.float()))
        return features, classes

    def load_classifiers(self, kfold):
        # load the given fold of all the classifiers
        clfs = [[] for _ in range(self.N)]
        for i, classifier in enumerate(self.classifiers):
            if isinstance(classifier, (EOG_double1dcnn_cv, EOG_1dcnn_cv)):
                clf = copy.deepcopy(classifier)
                assert self.K == clf.K, 'All classifiers must have the same number of folds, so that the splits coincide'
                clf = clf.load_epoch(clf.savepath, self.epochs[i], kfold)
                # if at some point I include sth diff than 1D-CNN, revise this line
                clf.eval()
                clfs[i] = clf
            elif isinstance(classifier, feature_loader):
                clfs[i] = classifier.to(self.device)  # no need to load anything
            else:
                print('Unknown classifier type:', type(classifier))
        self.loaded_clfs = clfs
        print('Fold %d loaded' %kfold)
        self.loaded_fold = kfold
        self.current_fold = kfold