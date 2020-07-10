"""Creates a base structure for any of the nets (CNN, ensemble, LSTM...) with cross-validation and implements
    many methods that are generizable for all of them (trainloop, evaluate, saving and loading, plotting..."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import time
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
import seaborn as sb

class base_EOGnn_cv(nn.Module):
    def __init__(self, K, dropout=True, dropout_prob=0.5, lr=0.0001, output_classes=5,
                 class_values=None, class_weights=None, labels=None,
                 use_GPU=True, savepath='D:/TFM/trained_models/'):
        """
        :param K: (int) number of cross-validation folds. Must be the same that K of the dataloader.
        :param dropout: (bool) whether to use dropout
        :param dropout_prob: (float) if using dropout, probability of dropping any given neuron
        :param lr: (float) learning rate to use in the gradient-based optimizer
        :param output_classes: (int) how many output classes there are. 5 (W, REM, N1, N2, N3) is the default, and should only
        be changed if for some reason some classes are to be merged (all NREM sleep stages together, for instance)
        :param class_values: (list of int) the values of the classes in the target vector (the hypnogram).
        :param class_weights: (list of numbers) whether to assing different emphasis to samples of different classes
        :param labels: (list of str) name of each class, to be used for plotting
        :param use_GPU: (bool) whether to use GPU to accelerate training
        :param savepath: (str) path where networks parameters are to be saved during training

        Initialize important parameters that are to be used by all child classes.
        """
        super().__init__()

        if class_values is None:
            class_values = [0, 1, 2, 3, 4]
        if class_weights is None:
            class_weights = [1, 1, 1, 1, 1]
        elif class_weights == 'balanced':
            class_weights = [1, 10, 0.8, 2, 3]
        if labels is None:
            labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

        # output parameters (later this will be removed)
        self.output_classes = output_classes
        self.class_values = class_values
        assert self.output_classes == len(self.class_values), \
            "The number of output classes and the length of class values array must have the same dimension"
        self.labels = labels
        self.current_seq_step = 0
        self.current_epoch = 0
        self.K = K  # how many cross-validation folds our NN is going to have

        # additional parameters
        if savepath[-1] is '/': savepath = savepath[:-1]  # remove the final /. It will be appended later.
        self.savepath = savepath
        if os.path.isdir(savepath) is False: os.mkdir(savepath)
        self.lr = lr
        self.criterion = nn.NLLLoss(weight=torch.from_numpy(np.asarray(class_weights)).float())
        # includes class balancing through weighting

        self.dropout_bool = dropout
        self.dropout_prob = dropout_prob
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.use_GPU = use_GPU
        if self.use_GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.to(self.device)
            else:
                self.device = torch.device("cpu")
                self.to(self.device)
                print('GPU is not available')
        else:
            self.device = torch.device("cpu")
            self.to(self.device)

        # to store the values (per kfold and per epoch)
        self.train_loss_during_training = [ [] for _ in range(self.K) ]
        self.train_acc_during_training  = [ [] for _ in range(self.K) ]
        self.train_balanced_acc         = [ [] for _ in range(self.K) ]
        self.train_f1perclass           = [ [] for _ in range(self.K) ]
        self.train_macrof1              = [ [] for _ in range(self.K) ]
        self.train_microf1              = [ [] for _ in range(self.K) ]
        self.train_kappa                = [ [] for _ in range(self.K) ]
        self.train_cm                   = [ [] for _ in range(self.K) ]

        self.valid_loss_during_training = [ [] for _ in range(self.K) ]
        self.valid_acc_during_training  = [ [] for _ in range(self.K) ]
        self.valid_balanced_acc         = [ [] for _ in range(self.K) ]
        self.valid_f1perclass           = [ [] for _ in range(self.K) ]
        self.valid_macrof1              = [ [] for _ in range(self.K) ]
        self.valid_microf1              = [ [] for _ in range(self.K) ]
        self.valid_kappa                = [ [] for _ in range(self.K) ]
        self.valid_cm                   = [ [] for _ in range(self.K) ]

        self.epoch_time                 = [ [] for _ in range(self.K) ]

    def create_optimizer(self, optimizer, params, lr=0):
        """
        :param optimizer: (str) type of optimizer to use
        :param params: (list of parameters) parameters over which the optimizer will operate
        :param lr: (float) learning rate to be applied
        :return: None

         Instantiate an optimizer, of the type specified, that operates over the specified parameters.
        """
        self.optimizer = optimizer.lower()  # set in lowercase to avoid problems
        if lr==0: lr = self.lr  # if not specified, learning rate is the one provided during object initialization

        if self.optimizer == 'adam':
            self.optim = optim.Adam(params, lr)
        elif self.optimizer == 'sgd':
            self.optim = optim.SGD(params, lr)
        elif self.optimizer == 'momentum':
            self.optim = optim.SGD(params, lr, momentum=0.9)
        elif self.optimizer == 'adagrad':
            self.optim = optim.Adagrad(params, lr)
        elif self.optimizer == 'adadelta':
            self.optim = optim.Adadelta(params)
        elif self.optimizer == 'rmsprop':
            self.optim = optim.RMSprop(params, lr)
        else:
            assert 1 == 0, 'Requested optimizer not recognized: ' + str(self.optimizer)

        return None

    def trainloop(self, dataloader, kfold, epochs, restart=True, batch_size=512, seq_len=100, shuffle=True, shuffling_seed='fixed'):
        # instead of defining one NN for each fold, it will take the number of folds from the dataloader
        # and train a different network for each, saving all the training data (acc, loss, pth file) for all the folds

        assert self.K == dataloader.K, "Dataloader and network must have the same number of cross-validation folds"

        self.current_fold = kfold

        if restart is True:
            self.current_epoch = -1  # NOW EPOCHS START IN ZERO
            # reinitialize the parameters of the network, loading the initial state_dict
            state_dict = torch.load(self.savepath + '/initialstate.pth')
            self.load_state_dict(state_dict)

            if self.type == 'ensemble':
                self.load_classifiers(self.current_fold)
            elif self.type == 'LSTM_ontop':
                self.load_classifier(kfold)
                if self.classifier.type != 'feature_loader':
                    # if it has an "evaluate" method (all but feature loader), use it to get a guess of how good is the CNN, which is the base for our LSTM
                    print('Classifier accuracy over validation set: %.3f' % self.classifier.evaluate(dataloader,
                           self.current_fold, validation=True, makeplots=False)[1])

        for e in range(epochs):
            t_start_epoch = time.time()
            self.current_epoch += 1
            self.current_seq_step = 0
            epoch_loss = 0.

            self.train()

            if shuffling_seed is 'fixed':
                seed = e  # allow to always use a different batch division each epoch, but always the same
            else:
                seed = np.random.randint(100)  # or let it be random

            # Normal training
            if self.type == 'EOG_1dcnn_cv' or self.type == 'ensemble':
                """All these classes work by batches, and optimization is performed once per batch"""
                # if validation set has been created, validation data will not be included in the batches
                N = len(dataloader.hypnogram_cv_train[kfold]) // batch_size
                assert N > 0, "Batch size is larger than the training set and therefore validation is not possible. Please reduce the batch size."
                generator = dataloader.batch_generator(kfold, batch_size, shuffle=shuffle, shuffling_seed=seed)
                lenh = batch_size * N
                topclass = np.zeros((lenh, 1))
                fullhyp = np.zeros((lenh, 1))
                current_pos = 0
                for data in generator:
                    self.current_seq_step += 1
                    torch_data = [torch.from_numpy(data[i]).float().to(self.device) for i in range(len(data))]
                    hyp = torch.from_numpy(data[1]).to(self.device).long()
                    _, classes = self.forward(torch_data)

                    self.optim.zero_grad()
                    loss = self.criterion(classes, hyp)
                    loss.backward()
                    self.optim.step()
                    epoch_loss += loss.item()

                    _, tc = classes.topk(1, dim=1)  # find the position (and label) for the most probable label for each datapoint
                    topclass[current_pos:current_pos + len(tc)] = tc.detach().cpu().numpy()
                    fullhyp[current_pos: current_pos + len(tc)] = hyp.view(-1, 1).detach().cpu().numpy()
                    current_pos += len(tc)

            elif self.type == 'LSTM_ontop':
                current_batch_size = int(batch_size + torch.randint(low=-2, high=2+1, size=(1,)))
                N = len(dataloader.hypnogram_cv_train[kfold]) // current_batch_size
                assert N > seq_len, "Batch size * seq_len is larger than the training set and therefore training is not possible. Please reduce the batch size."
                print(str(N//seq_len) + 'is the number of available training batch*seqlen')
                generator = dataloader.batch_generator(kfold, current_batch_size, shuffle=False)
                lenh = batch_size * (N//seq_len) * seq_len
                topclass = np.zeros((lenh, 1))
                fullhyp = np.zeros((lenh, 1))
                while self.current_seq_step + seq_len < N: # while there is room for one more sequence
                    hyp, fullout = self.LSTM_get_fullout_features(current_batch_size, seq_len, generator)

                    self.train()
                    self.optim.zero_grad()
                    _, lstm_classes = self.forward(fullout.float())
                    lstm_classes = lstm_classes.view(-1, self.output_classes)
                    hyp = hyp.view(-1, ).long()
                    loss = self.criterion(lstm_classes, hyp)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                    self.optim.step()
                    epoch_loss += loss.item()
                    _, tc = lstm_classes.topk(1, dim=-1)  # find the position (and label) for the most probable label for each datapoint
                    topclass[self.current_seq_step - seq_len: self.current_seq_step]= tc.detach().cpu().numpy()
                    fullhyp[self.current_seq_step - seq_len: self.current_seq_step] = hyp.view(-1, 1).detach().cpu().numpy()
            else:
                assert 0, "Unknown type of network"

            epoch_loss = epoch_loss / N
            epoch_acc = accuracy_score(fullhyp, topclass)
            epoch_bal_acc = balanced_accuracy_score(fullhyp, topclass)
            macrof1 = f1_score(fullhyp, topclass, average='macro') # unweighted avg of per-class F1 --> penalizes imbalance and bad performance in N1
            microf1 = f1_score(fullhyp, topclass, average='micro') # global TP --> does not take into account class imbalance
            perclassf1 = f1_score(fullhyp, topclass, average=None)
            kappa = cohen_kappa_score(fullhyp, topclass)
            cm = confusion_matrix(fullhyp, topclass, labels=self.class_values)
            # use acc and macro/micro F1 scores for plotting
            # cm and perclassF1 for deeper insight
            self.train_loss_during_training[kfold].append(epoch_loss)
            self.train_acc_during_training[kfold].append(epoch_acc)
            self.train_balanced_acc[kfold].append(epoch_bal_acc)
            self.train_f1perclass[kfold].append(perclassf1)
            self.train_macrof1[kfold].append(macrof1)
            self.train_microf1[kfold].append(microf1)
            self.train_kappa[kfold].append(kappa)
            self.train_cm[kfold].append(cm)

            # Testing loss and acc over evaluation set
            val_loss, val_acc, val_bal_acc, val_f1perclass, val_macrof1, val_microf1, val_kappa, val_cm \
                = self.evaluate(dataloader, kfold, validation=True, batch_size=batch_size, seq_len=seq_len)
            self.valid_loss_during_training[kfold].append(val_loss)
            self.valid_acc_during_training[kfold].append(val_acc)
            self.valid_balanced_acc[kfold].append(val_bal_acc)
            self.valid_f1perclass[kfold].append(val_f1perclass)
            self.valid_macrof1[kfold].append(val_macrof1)
            self.valid_microf1[kfold].append(val_microf1)
            self.valid_kappa[kfold].append(val_kappa)
            self.valid_cm[kfold].append(val_cm)

            # print epoch data and save it
            t_end_epoch = time.time()
            self.epoch_time[kfold].append(t_end_epoch - t_start_epoch)
            print("Epoch {:d}: \n\tTraining loss is {:.4f} and accuracy is {:.3f}. Kappa {:.3f}.".format(
                self.current_epoch, self.train_loss_during_training[kfold][-1],
                self.train_acc_during_training[kfold][-1],
                self.train_kappa[kfold][-1]))
            print("\tValidat  loss is {:.4f} and accuracy is {:.3f}. Kappa {:.3f}.".format(
                self.valid_loss_during_training[kfold][-1], self.valid_acc_during_training[kfold][-1],
                self.valid_kappa[kfold][-1]))
            print("\tElapsed time: {:.1f} seconds".format(self.epoch_time[kfold][-1]))
            self.save_during_training()
            print('Done!')

    def evaluate(self, dataloader, kfold, validation=False, batch_size=512, seq_len=100): # original, it does all at once, on GPU. Breaks for large dataset
        with torch.no_grad():
            self.eval()
            eval_loss = 0.
            generator = dataloader.batch_generator(kfold, batch_size, validation_set=validation, shuffle=False)
            if validation is True:  N = len(dataloader.hypnogram_cv_valid[kfold])//batch_size
            else:                   N = len(dataloader.hypnogram_cv_train[kfold])//batch_size
            assert N > 0, "Batch size is larger than the validation set and therefore validation is not possible. Please reduce the batch size."

            # Normal training
            if self.type == 'EOG_1dcnn_cv' or self.type == 'ensemble':
                """All these classes work by batches, and optimization is performed once per batch"""
                lenh = batch_size * N
                topclass = np.zeros((lenh, 1))
                fullhyp = np.zeros((lenh, 1))
                current_pos = 0
                for data in generator:
                    self.current_seq_step += 1
                    # torch_data = []
                    # for i in range(len(data)):
                    #     torch_data.append(torch.from_numpy(data[i]).float().to(self.device))
                    torch_data = [torch.from_numpy(data[i]).float().to(self.device) for i in range(len(data))]
                    hyp = torch.from_numpy(data[1]).to(self.device).long()
                    _, classes = self.forward(torch_data)

                    self.optim.zero_grad()
                    loss = self.criterion(classes, hyp)
                    eval_loss += loss.item()  # /batch_size

                    _, tc = classes.topk(1, dim=1)
                    topclass[current_pos:current_pos + len(tc)] = tc.detach().cpu().numpy()
                    fullhyp[current_pos: current_pos + len(tc)] = hyp.view(-1, 1).detach().cpu().numpy()
                    current_pos += len(tc)

            elif self.type == 'LSTM_ontop':
                print(str(N // seq_len) + 'is the number of available validation batch*seqlen')
                lenh = batch_size * (N // seq_len) * seq_len
                topclass = np.zeros((lenh, 1))
                fullhyp = np.zeros((lenh, 1))
                while self.current_seq_step + seq_len < N:  # while there is room for one more sequence
                    hyp, fullout = self.LSTM_get_fullout_features(batch_size, seq_len, generator)
                    _, lstm_classes = self.forward(fullout.float())
                    lstm_classes = lstm_classes.view(-1, self.output_classes)
                    hyp = hyp.view(-1, ).long()
                    loss = self.criterion(lstm_classes, hyp)
                    eval_loss += loss.item()
                    _, tc = lstm_classes.topk(1, dim=-1)
                    topclass[self.current_seq_step - seq_len: self.current_seq_step] = tc.detach().cpu().numpy()
                    fullhyp[self.current_seq_step - seq_len: self.current_seq_step] = hyp.view(-1, 1).detach().cpu().numpy()

            else:
                assert 0, "Unknown type of network"

            eval_loss = eval_loss / N
            eval_acc = accuracy_score(fullhyp, topclass)
            eval_bal_acc = balanced_accuracy_score(fullhyp, topclass)
            eval_macrof1 = f1_score(fullhyp, topclass, average='macro')  # unweighted avg of per-class F1 --> penalizes imbalance and bad performance in N1
            eval_microf1 = f1_score(fullhyp, topclass, average='micro')  # global TP --> does not take into account class imbalance
            eval_perclassf1 = f1_score(fullhyp, topclass, average=None)
            eval_kappa = cohen_kappa_score(fullhyp, topclass)
            eval_cm = confusion_matrix(fullhyp, topclass, labels=self.class_values)
        return eval_loss, eval_acc, eval_bal_acc, eval_perclassf1, eval_macrof1, eval_microf1, eval_kappa, eval_cm

    def predict_fullsubject(self, dataloader, subject):
        subject = int(subject)
        self.eval()
        data = (dataloader.pEOG[subject], dataloader.hypnogram[subject], np.zeros((len(dataloader.pEOG[subject]),2,5)), #dataloader.MSE[subject],
                dataloader.PSD[subject], dataloader.PSD_highres[subject], dataloader.spect[subject],
                dataloader.stats[subject], dataloader.AR_coefs[subject])

        torch_data = [torch.from_numpy(data[i]).float().to(self.device) for i in range(len(data))]

        with torch.no_grad():

            if self.type == 'EOG_1dcnn_cv' or self.type == 'ensemble':
                features, classes = self.forward(torch_data)

            elif self.type == 'LSTM_ontop':
                features, _ = self.classifier.forward(torch_data)
                # features, cnn_classes = self.classifier.forward(torch_data)
                # if self.use_features:
                _, classes = self.forward(features.unsqueeze(0).float())
                # else: _, classes = self.forward(cnn_classes.unsqueeze(0).float())
            else:
                assert 0, "Unknown type of network"

            classes = classes.view(-1, self.output_classes)
            _, tc = classes.topk(1, dim=-1)  # find the position (and label) for the most probable label for each datapoint
            topclass = tc.detach().cpu().numpy()

        return topclass

    def evaluate_all_validation_subjects(self, dataloader, best_epoch, makeprints=False):
        acc = []
        macrof1 = []
        kappa = []
        corr = []

        for k in range(dataloader.K):

            # LOAD UNDERLYING FEATURE EXTRACTOR(S) (if any)
            # print(self.type)
            self.current_fold = k
            if self.type == 'EOG_1dcnn_cv':
                pass
            elif self.type == 'ensemble':
                self.load_classifiers(k)
            elif self.type == 'LSTM_ontop':
                self.load_classifier(k)
            else:
                assert 0, "Unknown type of network"

            # LOAD THE CORRESPONDING FOLD AND EPOCH
            self.load_epoch(self.savepath, best_epoch, k)

            validation_subjects = dataloader.validation_subjects_cv[k]
            for i, subject in enumerate(validation_subjects):
                topclass = self.predict_fullsubject(dataloader, subject)
                hypnogram = dataloader.hypnogram[subject].reshape(-1, 1)

                acc.append(accuracy_score(hypnogram, topclass))
                macrof1.append(f1_score(hypnogram, topclass, average='macro'))
                kappa.append(cohen_kappa_score(hypnogram, topclass))
                corr.append(np.corrcoef(self.reformat_hypnogram(hypnogram).squeeze(),
                                        self.reformat_hypnogram(topclass).squeeze() )[0, 1])

        if makeprints:
            print('Accuracy: \t\t%.1f +- %.1f %%' % (np.mean(acc) * 100, np.std(acc) * 100))
            print('Macro F1: \t\t%.1f +- %.1f %%' % (np.mean(macrof1) * 100, np.std(macrof1) * 100))
            print('Kappa: \t\t\t{:.3f} +- {:.3f}'.format(np.mean(kappa), np.std(kappa)))
            print('Correlation: \t\t{:.3f} +- {:.3f}'.format(np.mean(corr), np.std(corr)))

        return acc, macrof1, kappa, corr

    def save_initial_state(self):
        # save the initial parameters, so we can reinitialize the network for each fold
        savename = '{0}/initialstate.pth'.format(self.savepath)
        torch.save(self.state_dict(), savename)

    def save_during_training(self):

        # save weights (state_dict)
        savename = '{0}/epoch{1:d}_cv{2:d}.pth'.format(self.savepath, self.current_epoch, self.current_fold)
        torch.save(self.state_dict(), savename)

        # save training data (acc and loss over training) for each epoch
        my_vars = [self.current_epoch,

                   self.train_loss_during_training,
                   self.train_acc_during_training,
                   self.train_balanced_acc,
                   self.train_f1perclass,
                   self.train_macrof1,
                   self.train_microf1,
                   self.train_kappa,
                   self.train_cm,

                   self.valid_loss_during_training,
                   self.valid_acc_during_training,
                   self.valid_balanced_acc,
                   self.valid_f1perclass,
                   self.valid_macrof1,
                   self.valid_microf1,
                   self.valid_kappa,
                   self.valid_cm,

                   self.epoch_time]

        # filename = self.savepath + '/training_information__epoch' + str(self.current_epoch) + '_cv' + str(self.current_fold) + '.trainrecord'
        # with open(filename, 'wb') as f:
        #     pickle.dump(my_vars, f)

        # # save a general training data file (overwritten each iteration)
        # changed recently, review if needed
        filename = self.savepath + '/last_training_information.trainrecord'   # keep as .trainrecord so that it is stored over GitHub
        with open(filename, 'wb') as f:
            pickle.dump(my_vars, f)

    def load_epoch(self, path, epoch, kfold, include_criterion_weight=True):
        if path[-1] is '/': path = path[:-1]  # remove the final /. It will be appended later.
        # state_dict = torch.load(path + '/epoch' + str(epoch) + '_cv' + str(kfold) + '.pth')
        state_dict = torch.load(path + '/epoch' + str(epoch) + '_cv' + str(kfold) + '.pth', map_location=torch.device(self.device))
        if include_criterion_weight:
            if "criterion.weight" not in state_dict: # ensure compatibility of old files without class weights
                state_dict["criterion.weight"] = torch.from_numpy(np.asarray([1,1,1,1,1])).float()
        self.load_state_dict(state_dict)

        # changed recently, review if needed
        # filename = path + '/training_information__epoch' + str(epoch) + '_cv' + str(kfold) + '.trainrecord'
        filename = path + '/last_training_information.trainrecord'
        if not os.path.isfile(filename):  # for old trainings (before 29th June)
            filename = path + '/training_information__epoch' + str(epoch) + '_cv' + str(kfold) + '.trainrecord'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if len(data) == 16: # without kappa, for backward compatibility
                [self.current_epoch,

                 self.train_loss_during_training,
                 self.train_acc_during_training,
                 self.train_balanced_acc,
                 self.train_f1perclass,
                 self.train_macrof1,
                 self.train_microf1,
                 self.train_cm,

                 self.valid_loss_during_training,
                 self.valid_acc_during_training,
                 self.valid_balanced_acc,
                 self.valid_f1perclass,
                 self.valid_macrof1,
                 self.valid_microf1,
                 self.valid_cm,

                 self.epoch_time] = data
            elif len(data) == 18:  # with kappa
                [self.current_epoch,

                 self.train_loss_during_training,
                 self.train_acc_during_training,
                 self.train_balanced_acc,
                 self.train_f1perclass,
                 self.train_macrof1,
                 self.train_microf1,
                 self.train_kappa,
                 self.train_cm,

                 self.valid_loss_during_training,
                 self.valid_acc_during_training,
                 self.valid_balanced_acc,
                 self.valid_f1perclass,
                 self.valid_macrof1,
                 self.valid_microf1,
                 self.valid_kappa,
                 self.valid_cm,

                 self.epoch_time] = data

            else:
                assert 0, "Trainrecord file has an unknown amount of content"
        return self

    def plot_training_information(self): # invoke when fully trained (or last epoch loaded)
        equallength = True
        minlength = len(self.epoch_time[0])
        for k in range(1, self.K):  # check that length is the same for all of the trainings
            if len(self.epoch_time[k]) < minlength:
                equallength = False
                minlength = len(self.epoch_time[k])

        if equallength is False: print("Not all trainings have the same length, so the minimum length will be used")

        # if equallength:
        L = minlength  # len(self.train_loss_during_training[0])
        full_train_loss = np.asarray(self.train_loss_during_training)[:,:L]
        full_valid_loss = np.asarray(self.valid_loss_during_training)[:,:L]
        full_train_acc  = np.asarray(self.train_acc_during_training)[:,:L]
        full_valid_acc  = np.asarray(self.valid_acc_during_training)[:,:L]

        # full_train_loss = np.zeros((self.K, L))
        # full_valid_loss = np.zeros((self.K, L))
        # full_train_acc = np.zeros((self.K, L))
        # full_valid_acc = np.zeros((self.K, L))
        full_valid_cm = np.zeros((self.K, self.output_classes, self.output_classes))

        # for k in range(self.K):
        #     full_train_loss[k, :] = self.train_loss_during_training[k][:L]
        #     full_valid_loss[k, :] = self.valid_loss_during_training[k][:L]
        #     full_train_acc[k, :] = self.train_acc_during_training[k][:L]
        #     full_valid_acc[k, :] = self.valid_acc_during_training[k][:L]

        mean_train_loss = np.mean(full_train_loss, axis=0)
        mean_valid_loss = np.mean(full_valid_loss, axis=0)
        best_epoch = int(np.argmin(mean_valid_loss))
        mean_train_acc = np.mean(full_train_acc, axis=0)
        mean_valid_acc = np.mean(full_valid_acc, axis=0)
        for k in range(self.K):
            full_valid_cm[k, :, :] = self.valid_cm[k][int(best_epoch)]
        sum_valid_cm = np.sum(full_valid_cm, axis=0)

        std_train_loss = np.std(full_train_loss, axis=0)
        std_valid_loss = np.std(full_valid_loss, axis=0)
        std_train_acc = np.std(full_train_acc, axis=0)
        std_valid_acc = np.std(full_valid_acc, axis=0)

        # plots of (mean) loss and acc, both for train and validation sets
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        x = np.arange(len(mean_train_acc))
        ax1.plot(x, mean_train_loss, 'r.-', label='Training loss ($\pm$std)')
        ax1.fill_between(x, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, facecolor="r", alpha=0.5)
        ax1.plot(x, mean_valid_loss, 'b.-', label='Validation loss ($\pm$std)')
        ax1.fill_between(x, mean_valid_loss - std_valid_loss, mean_valid_loss + std_valid_loss, facecolor="b", alpha=0.5)
        ax1.legend()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss during training')

        ax2.plot(x, mean_train_acc, 'r.-', label='Training acc ($\pm$std)')
        ax2.fill_between(x, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, facecolor="r", alpha=0.5)
        ax2.plot(x, mean_valid_acc, 'b.-', label='Validation acc ($\pm$std)')
        ax2.fill_between(x, mean_valid_acc - std_valid_acc, mean_valid_acc + std_valid_acc, facecolor="b", alpha=0.5)
        ax2.legend(loc='lower right')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel("Accuracy")
        ax2.set_title('Accuracy during training')
        ax2.set_ylim(0,1)

        plt.savefig(self.savepath + '/fulltraining.png')
        plt.show()

        # print the best epochs in terms of loss and accuracy
        best_epoch_acc = int(np.argmax(mean_valid_acc))
        print('Epoch %d has the highest validation accuracy (%.1f +- %.1f %%) with validation loss %.3f +- %.3f. '
              %(best_epoch_acc,
                mean_valid_acc[best_epoch_acc]*100,
                std_valid_acc[best_epoch_acc]*100,
                mean_valid_loss[best_epoch_acc],
                std_valid_loss[best_epoch_acc]))
        print('Epoch %d has the lowest validation loss (%.3f +- %.3f) with accuracy %.1f +- %.1f %%. \nTime taken to train (aprox): %.1f seconds.'
              % (best_epoch,
                 mean_valid_loss[best_epoch],
                 std_valid_loss[best_epoch],
                 mean_valid_acc[best_epoch]*100,
                 std_valid_acc[best_epoch]*100,
                 (np.sum(self.epoch_time[1][:best_epoch+1]) * 5) ))
        if len(self.valid_kappa[0])>0: self.plot_kappa(epoch_to_print=best_epoch)  # only display kappa if it is available
        self.plot_f1_score('macro', epoch_to_print=best_epoch)

        # display the confusion matrix of last epoch, summed over all folds (so of all subjects)
        # only display the one of validation. Maybe add the training one if it is appropiate.
        self.plot_cm(sum_valid_cm, 'Normalized confusion matrix of all validation sets \n at epoch '+str(best_epoch),
                'validationCM_fulltraining', savepath=self.savepath, labels=self.labels)

        return self, mean_valid_loss, mean_valid_acc

    def plot_kappa(self, epoch_to_print=0):
        equallength = True
        minlength = len(self.epoch_time[0])
        for k in range(1, self.K):  # check that length is the same for all of the trainings
            if len(self.epoch_time[k]) < minlength:
                equallength = False
                minlength = len(self.epoch_time[k])

        if equallength is False: print("Not all trainings have the same length, so the minimum length will be used")
        L = minlength  # len(self.train_loss_during_training[0])
        full_train_kappa = np.zeros((self.K, L))
        full_valid_kappa = np.zeros((self.K, L))
        for k in range(self.K):
            full_train_kappa[k, :] = self.train_kappa[k][:L]
            full_valid_kappa[k, :] = self.valid_kappa[k][:L]
        mean_train_kappa = np.mean(full_train_kappa, axis=0)
        mean_valid_kappa = np.mean(full_valid_kappa, axis=0)
        std_train_kappa = np.std(full_train_kappa, axis=0)
        std_valid_kappa = np.std(full_valid_kappa, axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        x = np.arange(len(mean_train_kappa))
        ax1.axis('off')

        ax2.plot(x, mean_train_kappa, 'rx-', label='Training kappa ($\pm$std)')
        ax2.fill_between(x, mean_train_kappa - std_train_kappa, mean_train_kappa + std_train_kappa, facecolor="r", alpha=0.5)
        ax2.plot(x, mean_valid_kappa, 'bx-', label='Validation kappa ($\pm$std)')
        ax2.fill_between(x, mean_valid_kappa - std_valid_kappa, mean_valid_kappa + std_valid_kappa, facecolor="b", alpha=0.5)
        ax2.legend(loc='lower right')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel("Cohen's kappa")
        ax2.set_title('Kappa evolution during training')
        ax2.set_ylim(0, 1)

        print('Epoch %d validation kappa: %.3f +- %.3f.'
              % (epoch_to_print,
                 mean_valid_kappa[epoch_to_print],
                 std_valid_kappa[epoch_to_print]))

    def plot_f1_score(self, whichf1score='macro', epoch_to_print=None):
        equallength = True
        for k in range(self.K - 1):  # check that length is the same for all of the trainings
            if len(self.epoch_time[k]) != len(self.epoch_time[k + 1]): equallength = False;

        if equallength is False: print("Not all trainings have the same length, so it is not possible to plot the results")

        if equallength:
            assert whichf1score is 'macro' or whichf1score is 'micro' \
                   or whichf1score is 'W'  or whichf1score is 'N1' \
                   or whichf1score is 'N2' or whichf1score is 'N3' \
                   or whichf1score is 'SWS' or whichf1score is 'REM'\
                   or whichf1score is 'all', 'That F1 category does not exist. Please, use either "all", "macro", "micro", or a sleep stage'
            L = len(self.train_loss_during_training[0])
            mylist_train = np.zeros((self.K, L))
            mylist_valid = np.zeros((self.K, L))
            if whichf1score is 'macro':
                for k in range(self.K):
                    mylist_train[k,:] = self.train_macrof1[k]
                    mylist_valid[k,:] = self.valid_macrof1[k]
                mytitle = 'Evolution of Macro F1 score'
            elif whichf1score is 'micro':
                for k in range(self.K):
                    mylist_train[k,:] = self.train_microf1[k]
                    mylist_valid[k,:] = self.valid_microf1[k]
                mytitle = 'Evolution of Micro F1 score'
            elif whichf1score is 'W':
                for k in range(self.K):
                    for e in range(L):
                        mylist_train[k,e] = self.train_f1perclass[k][e][0] # check this works
                        mylist_valid[k,e] = self.valid_f1perclass[k][e][0]
                mytitle = 'Evolution of Wake stage F1 score'
            elif whichf1score is 'N1':
                for k in range(self.K):
                    for e in range(L):
                        mylist_train[k,e] = self.train_f1perclass[k][e][1] # check this works
                        mylist_valid[k,e] = self.valid_f1perclass[k][e][1]
                mytitle = 'Evolution of N1 stage F1 score'
            elif whichf1score is 'N2':
                for k in range(self.K):
                    for e in range(L):
                        mylist_train[k,e] = self.train_f1perclass[k][e][2] # check this works
                        mylist_valid[k,e] = self.valid_f1perclass[k][e][2]
                mytitle = 'Evolution of N2 stage F1 score'
            elif whichf1score is 'N3' or whichf1score is 'SWS':
                for k in range(self.K):
                    for e in range(L):
                        mylist_train[k,e] = self.train_f1perclass[k][e][3] # check this works
                        mylist_valid[k,e] = self.valid_f1perclass[k][e][3]
                mytitle = 'Evolution of N3 (SWS) stage F1 score'
            elif whichf1score is 'REM':
                for k in range(self.K):
                    for e in range(L):
                        mylist_train[k,e] = self.train_f1perclass[k][e][4] # check this works
                        mylist_valid[k,e] = self.valid_f1perclass[k][e][4]
                mytitle = 'Evolution of REM stage F1 score'
            elif whichf1score is 'all':
                self.plot_f1_score('macro')
                self.plot_f1_score('micro')
                print('Individual sleep stages:')
                self.plot_f1_score('W')
                self.plot_f1_score('N1')
                self.plot_f1_score('N2')
                self.plot_f1_score('N3')
                self.plot_f1_score('REM')
                return

            mean_train = np.mean(mylist_train, axis=0)
            std_train  = np.std(mylist_train, axis=0)
            mean_valid = np.mean(mylist_valid, axis=0)
            std_valid  = np.std(mylist_valid, axis=0)

            if epoch_to_print is not None:
                print('The mean validation '+ whichf1score + ' F1 score at epoch '+str(epoch_to_print)+' is '
                      +str(np.round(mean_valid[epoch_to_print], 3)*100)+' +- '+str(np.round(std_valid[epoch_to_print],3)*100)+' %')

            fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
            x = np.arange(len(mean_train))
            ax1.plot(x, mean_train, 'r.-', label='Training F1 ($\pm$std)')
            ax1.fill_between(x, mean_train - std_train, mean_train + std_train, facecolor="r", alpha=0.5)
            ax1.plot(x, mean_valid, 'b.-', label='Validation F1 ($\pm$std)')
            ax1.fill_between(x, mean_valid - std_valid, mean_valid + std_valid, facecolor="b", alpha=0.5)
            ax1.legend()
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('F1 score')
            ax1.set_title(mytitle)
            ax1.set_ylim(0, 1)
            plt.show()

    def plot_cm_at_epoch(self, epoch, labels=None):
        if labels is None:
            labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        for k in range(self.K):
            assert len(self.train_cm[k]) >= epoch, 'That epoch does not exist in fold ' + str(k)

        # full_train_cm = np.zeros((self.K, self.output_classes, self.output_classes))
        full_valid_cm = np.zeros((self.K, self.output_classes, self.output_classes))
        for k in range(self.K):
            # full_train_cm[k, :, :] = self.train_cm[k][epoch-1]
            full_valid_cm[k, :, :] = self.valid_cm[k][epoch]
        # sum_train_cm = np.sum(full_train_cm, axis=0)
        sum_valid_cm = np.sum(full_valid_cm, axis=0)
        # title_train = 'Normalized confussion matrix \nof training sets at epoch ' + str(epoch)
        title_valid = 'Normalized confussion matrix \nof all validation sets at epoch ' + str(epoch)

        # plot_cm(sum_train_cm, title_train, 'trainingCM_epoch'+str(epoch), savepath=self.savepath)
        self.plot_cm(sum_valid_cm, title_valid, 'validationCM_epoch'+str(epoch), savepath=self.savepath,labels=labels)

    @staticmethod
    def plot_cm(cm, title, savename=None, savepath=None, labels=None):
        if labels is None:
            labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        counts_per_class = cm.sum(axis=1)  # how much of each class is there (real classes)
        counts_per_class[counts_per_class == 0] = 1  # to avoid division by zero
        normcm = (cm.T / counts_per_class).T  # normalize by real classes

        cmap = 'gist_gray'  # 'coolwarm'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 5]})
        norm_counts_per_class = (counts_per_class / np.sum(counts_per_class)).reshape(-1, 1)
        sb.heatmap(norm_counts_per_class, ax=ax1, annot=True, fmt='.2f', cmap=cmap, xticklabels=[], yticklabels=labels,
                   cbar=False, vmin=0, vmax=0.5)
        sb.heatmap(normcm, ax=ax2, annot=True, fmt='.2f', cmap=cmap, xticklabels=labels, linewidths=0.05, cbar=False,
                   vmin=0, vmax=1, yticklabels=labels)
        # ax1.set_ylabel('Human scoring')
        ax2.set_ylabel('Human scoring')
        ax2.set_xlabel('Automatic scoring')
        ax1.tick_params(axis='both', which='both', length=0)
        ax2.tick_params(axis='both', which='both', length=0)
        ax1.set_title('Relative density of\neach sleep stage')
        ax2.set_title(title)
        if savename is not None and savepath is not None:  # lets assume then both are strings
            plt.savefig(savepath + '/' + savename + '.png')
        plt.show()

    def plot_result_subject(self, dataloader, subject, topclass, label, start_idx=50, end_idx=250):
        # topclass can be either one single vector, or a list of vectors
        # correspondingly, label must one string, or a list of strings of the same length
        fig, ax = plt.subplots(1,1,figsize=(9, 5))
        ax.plot(self.reformat_hypnogram(dataloader.hypnogram[subject]), label='True hypnogram', linewidth=8)
        if type(topclass) is np.ndarray:  # single vector
            ax.plot(self.reformat_hypnogram(topclass), label=label)

        elif type(topclass) is list:  # list of vectors
            assert len(topclass) == len(label), "The amount of topclass and of labels must be the same"
            for i in range(len(topclass)):
                ax.plot(self.reformat_hypnogram(topclass[i]), label=label[i])

        ax.set_ylabel('Sleep Stage')
        ax.set_yticks([6, 7, 8, 9, 10])
        labels = ['Wake', 'REM', 'N1', 'N2', 'N3']
        labels.reverse()
        ax.set_yticklabels(labels)
        ax.set_xlabel('Sleep epochs')

        plt.xlim(start_idx, end_idx)
        plt.legend(loc='upper right')
        plt.show()
        return fig

    @staticmethod
    def boxplot_distribution(data_list, labels_list, metric_name='Accuracy'):
        """Creates a boxplot of the distribution of the acc (or any other metric) across different classifiers.
        :param data_list : a list in which each element is a list/ndarray of a metric for each subject
        :param labels_list : a list of labels to assign to each of the data arrays. Must have same length."""
        assert len(data_list) == len(labels_list), "Data and labels lists must have the same length"
        plt.boxplot(data_list, labels=labels_list)
        plt.title('Distribution of '+metric_name.lower()+' across different methods')
        plt.ylabel(metric_name)
        plt.ylim(0,1)
        plt.show()

    @staticmethod
    def reformat_hypnogram(hypnogram):
        # moves from order W - N1-3 - REM
        # to N3-1 - REM - W
        newhyp = np.zeros(hypnogram.shape)
        newhyp[hypnogram==0] = 10
        newhyp[hypnogram==4] = 9
        newhyp[hypnogram==1] = 8
        newhyp[hypnogram==2] = 7
        newhyp[hypnogram==3] = 6
        return newhyp

    def count_parameters(self, model=None):
        if model is None: model = self
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

