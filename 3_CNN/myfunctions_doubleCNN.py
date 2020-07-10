"""Implement two CNN of four convolutional layers each based in DeepSleepNet (Supratak et al. 2017)"""

import torch
from torch import nn
from myfunctions_base_EOGnn_cv import base_EOGnn_cv


class EOG_double1dcnn_cv(base_EOGnn_cv):
    def __init__(self, K=5, lr=0.0001, output_classes=5,
                 class_values=None, class_weights=None, labels=None,
                 optimizer='adam', mode='vanilla',
                 use_GPU=True, savepath='D:/GitHub/TFM/02_1DCNN_EOG/trained_models/' ):
        """
        :param mode: (str) either to use "vanilla" architecture that yields 2048 features, or "expanded_output", which
        yields a much larger number of features. Default: "vanilla" (it was observed that "expanded_output" did not
        improved results)

        All other parameters are to be passed to "base_EOGnn_cv", please check their specifications in that class definition.
        """


        super().__init__(K=K, dropout=False, dropout_prob=0, lr=lr, output_classes=output_classes,
                         class_values=class_values, class_weights=class_weights, labels=labels,
                         use_GPU=use_GPU, savepath=savepath)

        self.type = 'EOG_1dcnn_cv'  # also counts as EOG cnn for all methods

        # Now the architecture, it is divided in two dimensions (d1 and d2) with each two blocks (b1 and b2)
        # naming of the layers will follow that

        # Dimension 1 (Fs/2) Block 1
        if mode == "vanilla":
            d1_b1_maxpool = 8; d1_b2_maxpool = 4
            d1_b2_kernel  = 8; d1_b2_padding = 0

            d2_b1_maxpool = 4; d2_b2_maxpool = 2
            d2_b2_kernel  = 6; d2_b2_padding = 0

            self.outputsize = 2048  # 128*(10+6)

        elif mode == "expanded_output":
            d1_b1_maxpool = 4; d1_b2_maxpool = 2
            d1_b2_kernel  = 6; d1_b2_padding = 3

            d2_b1_maxpool = 2; d2_b2_maxpool = 1
            d2_b2_kernel  = 4; d2_b2_padding = 2

            self.outputsize = 15360  # 128*(63+57) # not 58
        else:
            assert 0, 'Input a correct mode!!'

        self.d1_b1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=25, stride=3, padding=0),
            # 2 input (left and right) and 64 output channels
            # input dim = Fs/2 = 25 and stride = Fs/16 = 3.125
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(d1_b1_maxpool),
            nn.Dropout(p=0.5)
        )

        # Dimension 1 (Fs/2) Block 2
        self.d1_b2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=d1_b2_kernel, padding=d1_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=d1_b2_kernel, padding=d1_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=d1_b2_kernel, padding=d1_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(d1_b2_maxpool)
        )

        # Dimension 2 (4Fs) Block 1
        self.d2_b1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=200, stride=12, padding=0),
            # 2 input (left and right) and 64 output channels
            # input dim = 4Fs = 200 and stride = Fs/4 = 12
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(d2_b1_maxpool),
            nn.Dropout(p=0.5)
        )

        # Dimension 2 (4Fs) Block 2
        self.d2_b2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=d2_b2_kernel, padding=d2_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=d2_b2_kernel, padding=d2_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=d2_b2_kernel, padding=d2_b2_padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(d2_b2_maxpool)
        )

        # Classifier (linear layer)
        self.classifier_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # dropout before classifier (robustness)
            nn.Linear(in_features=self.outputsize, out_features=output_classes),
            nn.LogSoftmax(dim=1)
        )

        self.to(self.device)

        params_list = [{'params': self.d1_b1.parameters()},
                       {'params': self.d1_b2.parameters()},
                       {'params': self.d2_b1.parameters()},
                       {'params': self.d2_b2.parameters()},
                       {'params': self.classifier_layer.parameters()}]

        self.create_optimizer(optimizer, params=params_list)
        self.save_initial_state()

    def forward(self, data):
        x = data[0]
        x1 = self.d1_b1(x)  # N x 64 x 61 (instead of 62 ?)
        x1 = self.d1_b2(x1)

        x2 = self.d2_b1(x)  # N x 64 x 54 (instead of 55 ?)
        x2 = self.d2_b2(x2) # N x 64 x 57 (instead of 58 ?)

        features = torch.cat((x1.view(x1.shape[0], -1), x2.view(x1.shape[0], -1)),dim=1)
        # print('Shape of the features tensor', features.shape)

        classes = self.classifier_layer(features)
        # outputs "classes" with already the logsoftmax integrated.
        # print('Shape of classes', classes.shape)
        return features, classes
