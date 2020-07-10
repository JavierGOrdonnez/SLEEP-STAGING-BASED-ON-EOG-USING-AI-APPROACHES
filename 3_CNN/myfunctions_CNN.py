"""Implement a CNN of four convolutional layers based in DeepSleepNet (Supratak et al. 2017)"""

import numpy as np
from torch import nn
from myfunctions.myfunctions_base_EOGnn_cv import base_EOGnn_cv


class EOG_1dcnn_cv(base_EOGnn_cv):  # create the class with all the different parameters

    def __init__(self, input_dim, K,
                 input_stride=None, input_inchannels=2, input_outchannels=64, input_padding=0,
                 strides=None, conv_size=8, channels=None, padding=None,
                 dropout=True, maxpool_size=None, dropout_prob=0.5, lr=0.0001,
                 output_classes=5, class_values=None, class_weights=None, labels=None,
                 print_dims=False, optimizer='adam',
                 use_GPU=True, savepath='D:/TFM/trained_models/'):
        """
        :param input_dim: (int) size of the input kernel. Usually Fs/2 or 4Fs.

        Params "input_stride", "input_padding", "strides", etc offer the possibility of customizing the architecture of
        the CNN. Leave as default for Fs/2 CNN. For 4Fs, use conv_size=6, maxpool_size=[4,2], input_stride=Fs/4.

        :param print_dims: (bool) if True, print the output dimensions of each layer. Useful in order to fine-tune the
        architecture hyperparameters.

        All other parameters are to be passed to "base_EOGnn_cv", please check their specifications in that class definition.
        """


        super().__init__(K=K, dropout=dropout, dropout_prob=dropout_prob, lr=lr, output_classes=output_classes,
                         class_values=class_values, class_weights=class_weights, labels=labels,
                         use_GPU=use_GPU, savepath=savepath)

        self.type = 'EOG_1dcnn_cv'

        # architecture based on DeepSleepNet
        # Block 1: A first 1DCNN of given dim (small or large) kernel
        # Block 2: 3 succesive 1DCNN to extract features.
        # Batchnorm and ReLU after each 1DCNN in both blocks
        # Maxpool and dropout after each "block"
        # Finally a logsoftmax, as initially we will be using this for classification (later to be feed into LSTM)

        # Block 1 parameters
        self.input_dim = int(np.round(input_dim))  # input kernel size
        if input_stride is None: input_stride = input_dim / 8
        self.input_stride = int(np.round(input_stride))
        self.input_inchannels = input_inchannels  # input channels is always 2 (left and right EOG)
        self.input_outchannels = input_outchannels  # output channels of the first conv layer (block 1)
        self.input_padding = input_padding

        # Block 2 parameters
        if channels is None: channels = [128, 128, 128]  # number of output channels of 3 conv layers
        self.block2_channels = channels
        self.block2_convsize = conv_size  # conv kernel size to be applied in the 3 conv layers of block 2
        if strides is None: strides = [1, 1, 1]  # strides for the 3 conv layers of block 2
        self.block2_strides = strides
        if padding is None: padding = [0, 0, 0]
        self.block2_padding = padding

        # Common parameters
        if maxpool_size is None: maxpool_size = [8, 4]  # after each block
        self.maxpool_size = maxpool_size

        # Block 1 layer
        self.block1_conv = nn.Conv1d(self.input_inchannels, self.input_outchannels, self.input_dim, self.input_stride,
                                     self.input_padding)
        self.block1_bn = nn.BatchNorm1d(self.input_outchannels)
        self.block1_maxpool = nn.MaxPool1d(self.maxpool_size[0])

        # Block 2 layers
        self.block2_conv1 = nn.Conv1d(self.input_outchannels, self.block2_channels[0], self.block2_convsize,
                                      self.block2_strides[0], self.block2_padding[0])
        self.block2_bn1 = nn.BatchNorm1d(self.block2_channels[0])
        self.block2_conv2 = nn.Conv1d(self.block2_channels[0], self.block2_channels[1], self.block2_convsize,
                                      self.block2_strides[1], self.block2_padding[1])
        self.block2_bn2 = nn.BatchNorm1d(self.block2_channels[1])
        self.block2_conv3 = nn.Conv1d(self.block2_channels[1], self.block2_channels[2], self.block2_convsize,
                                      self.block2_strides[2], self.block2_padding[2])
        self.block2_bn3 = nn.BatchNorm1d(self.block2_channels[2])
        self.block2_maxpool = nn.MaxPool1d(self.maxpool_size[1])

        # provisional output layer
        self.outputsize_inputconv = np.floor(
            ((1500 + 2 * input_padding - (self.input_dim - 1) - 1) / self.input_stride + 1))
        if print_dims: print('Size of the output of the first convolution', self.outputsize_inputconv)
        self.outputsize_block1 = np.floor(self.outputsize_inputconv / self.maxpool_size[0])
        if print_dims: print('Size of the output of block 1 (after maxpooling)', self.outputsize_block1)
        self.outputsize_conv1 = np.floor(
            (self.outputsize_block1 + 2 * padding[0] - (conv_size - 1) - 1) / strides[0] + 1)
        if print_dims: print('Size of the output of the first convolution of block 2', self.outputsize_conv1)
        self.outputsize_conv2 = np.floor(
            (self.outputsize_conv1 + 2 * padding[1] - (conv_size - 1) - 1) / strides[1] + 1)
        if print_dims: print('Size of the output of the second convolution of block 2', self.outputsize_conv2)
        self.outputsize_conv3 = np.floor(
            (self.outputsize_conv2 + 2 * padding[2] - (conv_size - 1) - 1) / strides[2] + 1)
        if print_dims: print('Size of the output of the third convolution of block 2', self.outputsize_conv3)
        self.outputsize_block2 = np.floor(self.outputsize_conv3 / maxpool_size[1])
        if print_dims: print('Size of the output of block 2', self.outputsize_block2)
        self.outputsize = int(self.outputsize_block2 * self.block2_channels[2])
        if print_dims: print('Input to the final layer', self.outputsize)
        self.output_linear_layer = nn.Linear(int(self.outputsize_block2 * self.block2_channels[2]),
                                             int(self.output_classes))
        if print_dims: print('Size of the output of the linear layer', self.output_classes)

        self.to(self.device)
        self.create_optimizer(optimizer, params=self.parameters())
        self.save_initial_state()

    def forward(self, data):
        x = data[0]  # just need EOG signal

        # Block 1 layer
        x = self.block1_conv(x)
        x = self.block1_bn(x)  # batchnorm
        x = self.relu(x)

        x = self.block1_maxpool(x)
        if self.dropout_bool:
            x = self.dropout(x)

        # Block 2 layers
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.relu(x)

        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.relu(x)

        x = self.block2_conv3(x)
        x = self.block2_bn3(x)
        x = self.relu(x)

        x = self.block2_maxpool(x)
        x = x.view(x.shape[0], -1)  # shape 0 is batch size. All features of same sample are being flattened.
        # "x" is the processed features

        # apply provisional output layer
        if self.dropout_bool:
            y = self.dropout(x)
        y = self.output_linear_layer(y)
        classes = self.logsoftmax(y)

        return x, classes
