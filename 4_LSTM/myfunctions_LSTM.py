""" LSTM that uses the features extracted by an underlying classifier as input, and performs sequence learning over the
sequence of sleep states."""

import torch
from torch import nn
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

from myfunctions_base_EOGnn_cv import base_EOGnn_cv
from myfunctions_cnn import EOG_1dcnn_cv
from myfunctions_ensemble import ensemble
from myfunctions_doubleCNN import EOG_double1dcnn_cv

class LSTM_ontop(base_EOGnn_cv):

    def __init__(self, classifier, epoch_to_load=0, K=5, lr=1e-4, use_GPU=True,
                 output_classes=5, class_values=None, class_weights=None, labels=None,
                 hidden_size=100, bidirectional=True, num_layers=2,
                 dropout=True, dropout_prob=0.5, optimizer='Adam', clip=10, FC=False,
                 savepath='D:/GitHub/TFM/02_1DCNN_EOG/trained_models/'):
        """
        :param classifier: already trained/loaded classifier, with a "forward" method which returns (features, classes)
        In the case of ensembles, they do not need to be loaded, the function will do.
        :param epoch_to_load: when providing a EOG_cnn, provide the epoch we should load. For ensembles, the epochs are
        in the initialization of the ensemble already

        All other parameters are to be passed to "base_EOGnn_cv", please check their specifications in that class definition.
        """

        super().__init__(K=K, dropout=dropout, dropout_prob=dropout_prob, lr=lr, output_classes=output_classes,
                         class_values=class_values, class_weights=class_weights, labels=labels,
                         use_GPU=use_GPU, savepath=savepath)

        self.type = 'LSTM_ontop'

        self.classifier = classifier
        self.epoch_to_load = epoch_to_load
        if type(self.classifier) is EOG_1dcnn_cv or type(self.classifier) is EOG_double1dcnn_cv:
            assert self.epoch_to_load > 0, "An epoch to load greater than 0 must be provided"
        # self.use_features = use_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.clip = clip
        self.FullyConnected = FC

        self.lstm = nn.LSTM(input_size=self.classifier.outputsize, hidden_size=self.hidden_size, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout_prob, batch_first=True)

        print('Hidden size:', self.lstm.hidden_size)
        print('Input features size:', self.lstm.input_size)

        if self.bidirectional:
            self.classifier_layer = nn.Linear(2 * self.hidden_size, self.output_classes)
            if self.FullyConnected: self.fully_connected = nn.Linear(self.classifier.outputsize, 2 * self.hidden_size)
        else:
            self.classifier_layer = nn.Linear(self.hidden_size, self.output_classes)
            if self.FullyConnected: self.fully_connected = nn.Linear(self.classifier.outputsize, self.hidden_size)

        self.to(self.device)

        if self.FullyConnected:
            params_list = [{'params': self.lstm.parameters()},
                           {'params': self.fully_connected.parameters()},
                           {'params': self.classifier_layer.parameters()}]
        else:
            params_list = [{'params': self.lstm.parameters()},
                           {'params': self.classifier_layer.parameters()}]

        self.create_optimizer(optimizer, params=params_list)

        # save the initial parameters, so we can reinitialize the network for each fold
        self.save_initial_state()

    def forward(self, fullout):
        """
        Input: features from self.classifier, stacked in [batch_size, seq_len, feat_dim]
        """
        lstm_out = self.lstm(fullout)[0]
        if self.FullyConnected:
            fc_out = self.fully_connected(fullout)
            out = lstm_out + fc_out
        else:
            out = lstm_out
        classes = self.logsoftmax(self.classifier_layer(out))

        return out, classes

    def trainloop_persubject(self, dataloader, kfold, epochs, restart=True, subjects_per_batch=25):
        """ Instead of normal batching, we will be training by full subjects.
            More suited for sequence training."""
        if restart is True: self.restart_LSTM(dataloader, kfold)
        else: self.current_fold = kfold

        for e in range(epochs):

            t_start_epoch = time.time()
            self.current_epoch += 1

            # for each subject, do a forward, get the loss, and backpropagate.
            # Do not reset grads to zero, let them accumulate for "subjects_per_batch" to have some kind of batching
            total_n_subjects = len(dataloader.pEOG)
            n = 0  # keep number of subjects currently in this batch
            epoch_loss = 0.; epoch_acc = 0.; epoch_bal_acc = 0.; macrof1 = 0.; microf1 = 0.; kappa = 0.
            perclassf1 = np.zeros((self.output_classes,)); cm = np.zeros((self.output_classes, self.output_classes))
            self.current_seq_step = 0
            subjects = np.random.permutation(total_n_subjects)
            # randomly shuffles the order of the subjects in each epoch, to have some variety in training
            for ii in subjects:
                if ii not in dataloader.validation_subjects_cv[kfold]:
                    # print('Doing subject '+str(ii))
                    with torch.no_grad():
                        self.classifier.eval()
                        features, _ = self.LSTM_get_features_persubject(dataloader, ii)
                        hyp = torch.from_numpy(dataloader.hypnogram[ii]).to(self.device).long()

                    self.train()
                    self.classifier.eval()
                    _, lstm_classes = self.forward(features.unsqueeze(0).float())
                    lstm_classes = lstm_classes.view(-1, self.output_classes)
                    loss = self.criterion(lstm_classes, hyp)
                    loss.backward()  # perform backprop for this subject, let it accumulate
                    epoch_loss += loss.item()
                    _, tc = lstm_classes.topk(1, dim=-1)
                    topclass = tc.detach().cpu().numpy()
                    fullhyp = hyp.detach().cpu().numpy()

                    epoch_acc += accuracy_score(fullhyp, topclass)
                    epoch_bal_acc += balanced_accuracy_score(fullhyp, topclass)
                    macrof1 += f1_score(fullhyp, topclass, average='macro')
                    microf1 += f1_score(fullhyp, topclass, average='micro')
                    # perclassf1 += np.array(f1_score(fullhyp, topclass, average=None))
                    perclassf1 += f1_score(fullhyp, topclass, average=None)
                    kappa += cohen_kappa_score(fullhyp, topclass)
                    cm += confusion_matrix(fullhyp, topclass, labels=self.class_values)

                    n += 1

                    if (n % subjects_per_batch == 0) or \
                            n+len(dataloader.validation_subjects_cv[kfold]) == total_n_subjects:
                        # time to perform backprop: either reached "subjects_per_batch" or done all subjects left
                        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                        self.optim.step()
                        self.current_seq_step += 1
                        # print('Optimization step performed')
                        self.optim.zero_grad() # set gradients to zero, to start next batch

            self.train_loss_during_training[kfold].append(epoch_loss/n)
            self.train_acc_during_training[kfold].append(epoch_acc/n)
            self.train_balanced_acc[kfold].append(epoch_bal_acc/n)
            self.train_f1perclass[kfold].append(perclassf1/n)
            self.train_macrof1[kfold].append(macrof1/n)
            self.train_microf1[kfold].append(microf1/n)
            self.train_kappa[kfold].append(kappa/n)
            self.train_cm[kfold].append(cm)  # cm should not be divided by n right?

            # Testing loss and acc over evaluation set
            val_loss, val_acc, val_bal_acc, val_f1perclass, val_macrof1, val_microf1, val_kappa, val_cm = \
                self.evaluate_persubject(dataloader, kfold)
            # use standard batch size so that evaluation scores are always over the same data and allow comparison
            # as we will be having N * 1/K samples instead of N*(K-1/K), we reduce the batch size accordingly to have similar lengths
            self.valid_loss_during_training[kfold].append(val_loss)
            self.valid_acc_during_training[kfold].append(val_acc)
            self.valid_balanced_acc[kfold].append(val_bal_acc)
            self.valid_f1perclass[kfold].append(val_f1perclass)
            self.valid_macrof1[kfold].append(val_macrof1)
            self.valid_microf1[kfold].append(val_microf1)
            self.valid_kappa[kfold].append(val_kappa)
            self.valid_cm[kfold].append(val_cm)

            # testing per-subject on the evaluation subjects
            _, cnn_acc, lstm_acc = self.predict_all_validation_subjects(dataloader,kfold,makeprints=False)

            # print epoch data and save it
            t_end_epoch = time.time()
            self.epoch_time[kfold].append(t_end_epoch - t_start_epoch)
            print("Epoch %d: \n\tTraining loss is %.4f and accuracy is %.3f. Kappa %.3f." % (
                self.current_epoch, self.train_loss_during_training[kfold][-1],
                self.train_acc_during_training[kfold][-1], self.train_kappa[kfold][-1]))
            print("\tValidat  loss is %.4f and accuracy is %.3f. Kappa %.3f." % (
                self.valid_loss_during_training[kfold][-1], self.valid_acc_during_training[kfold][-1],
                self.valid_kappa[kfold][-1]))
            # print("\tValidation per-subject mean LSTM acc is %.3f." %(self.mean_lstm_acc[kfold][-1]))
            print("\tValidation per-subject mean LSTM acc is %.3f." %(np.mean(lstm_acc)))
            print("\tElapsed time: %.1f seconds" % self.epoch_time[kfold][-1])
            self.save_during_training()

    def evaluate_fullsubject(self, dataloader, subject, kfold, makeplots=False):
        # # load the underlaying classifier if necessary (to be in the correct fold)
        if kfold is not self.current_fold:
            print('Provided kfold %d is not the current fold %d' %(kfold, self.current_fold))
            self.load_classifier(self.current_fold)
        # the LSTM classifier is assumed to be the correct one (self.current_fold)
        subject = int(subject)
        self.eval()
        self.classifier.eval()

        features, classes = self.LSTM_get_features_persubject(dataloader, subject)

        cnn_classes = classes.view(-1, self.output_classes)
        _, tc = cnn_classes.topk(1, dim=-1)  # find the position (and label) for the most probable label for each datapoint
        cnn_topclass = tc.detach().cpu().numpy()

        _, lstm_classes = self.forward(features.unsqueeze(0).float())

        lstm_classes = lstm_classes.view(-1, self.output_classes)
        _, tc = lstm_classes.topk(1,dim=-1)  # find the position (and label) for the most probable label for each datapoint
        lstm_topclass = tc.detach().cpu().numpy()

        cnn_acc = accuracy_score(cnn_topclass, dataloader.hypnogram[subject])
        if makeplots: print('CNN accuracy:', cnn_acc)
        lstm_acc = accuracy_score(lstm_topclass, dataloader.hypnogram[subject])
        if makeplots: print('LSTM accuracy:', lstm_acc)
        cnn_f1 = f1_score(dataloader.hypnogram[subject], cnn_topclass, average=None)
        if makeplots: print('CNN F1 scores:', cnn_f1)
        lstm_f1 = f1_score(dataloader.hypnogram[subject], lstm_topclass, average=None)
        if makeplots: print('LSTM F1 scores:', lstm_f1)
        cnn_macrof1 = f1_score(dataloader.hypnogram[subject], cnn_topclass, average='macro')
        if makeplots: print('CNN F1 scores:', cnn_macrof1)
        lstm_macrof1 = f1_score(dataloader.hypnogram[subject], lstm_topclass, average='macro')
        if makeplots: print('LSTM F1 scores:', lstm_macrof1)
        if makeplots:
            self.plot_cm(confusion_matrix(dataloader.hypnogram[subject], cnn_topclass), 'CNN confusion matrix')
            self.plot_cm(confusion_matrix(dataloader.hypnogram[subject], lstm_topclass), 'LSTM confusion matrix')
            self.plot_result_subject(dataloader, subject, [cnn_topclass, lstm_topclass],
                                     ['CNN prediction', 'LSTM prediction'])
        scores = {'cnn':{'acc':cnn_acc,'f1':cnn_f1,'macrof1':cnn_macrof1},
                  'lstm':{'acc':lstm_acc,'f1':lstm_f1,'macrof1':lstm_macrof1}}

        return cnn_topclass, lstm_topclass, scores

    def evaluate_persubject(self, dataloader, kfold):
        """ A simple method inspired in "evaluate" and "predict_all_validation_subjects" to be used with
        "trainloop_persubject" """
        with torch.no_grad():
            validation_subjects = dataloader.validation_subjects_cv[kfold]
            self.eval()
            self.classifier.eval()
            eval_loss = 0.  ; eval_acc = 0. ; eval_bal_acc = 0. ; eval_macrof1 = 0. ; eval_microf1 = 0.; eval_kappa = 0.
            eval_perclassf1 = np.zeros((self.output_classes,))
            eval_cm = np.zeros((self.output_classes, self.output_classes))
            n = len(validation_subjects)
            for ii in validation_subjects:
                # print('Evaluating subject ' + str(ii))
                self.classifier.eval()
                features, _ = self.LSTM_get_features_persubject(dataloader, ii)
                hyp = torch.from_numpy(dataloader.hypnogram[ii]).to(self.device).long()

                _, lstm_classes = self.forward(features.unsqueeze(0).float())
                lstm_classes = lstm_classes.view(-1, self.output_classes)
                loss = self.criterion(lstm_classes, hyp)
                eval_loss += loss.item()
                _, tc = lstm_classes.topk(1, dim=-1)
                topclass = tc.detach().cpu().numpy()
                fullhyp = hyp.detach().cpu().numpy()
                eval_acc += accuracy_score(fullhyp, topclass)
                eval_bal_acc += balanced_accuracy_score(fullhyp, topclass)
                eval_macrof1 += f1_score(fullhyp, topclass, average='macro')
                eval_microf1 += f1_score(fullhyp, topclass, average='micro')
                eval_perclassf1 += np.array(f1_score(fullhyp, topclass, average=None))
                eval_kappa += cohen_kappa_score(fullhyp, topclass)
                eval_cm += confusion_matrix(fullhyp, topclass, labels=self.class_values)

        return eval_loss/n, eval_acc/n, eval_bal_acc/n, eval_perclassf1/n, eval_macrof1/n, eval_microf1/n, eval_kappa/n, eval_cm

    def LSTM_get_features_persubject(self, dataloader, ii):
        data = (dataloader.pEOG[ii], dataloader.hypnogram[ii], dataloader.MSE[ii],
                dataloader.PSD[ii], dataloader.PSD_highres[ii], dataloader.spect[ii],
                dataloader.stats[ii], dataloader.AR_coefs[ii])
        torch_data = [torch.from_numpy(data[i]).float().to(self.device) for i in range(len(data))]

        features, classes = self.classifier.forward(torch_data)

        return features, classes

    def LSTM_get_fullout_features(self, batch_size, seq_len, generator):
        # if self.use_features:
        fullout = torch.zeros((batch_size, seq_len, self.classifier.outputsize)).to(self.device)
        # else: fullout = torch.zeros((batch_size, seq_len, self.output_classes)).to(self.device)
        fullhyp = torch.zeros((batch_size, seq_len)).to(self.device)
        with torch.no_grad():
            self.classifier.eval()  # set in eval mode
            # for data in generator:
            for step in range(seq_len):
                data = generator.next()
                torch_data = []
                for i in range(len(data)):
                    torch_data.append(torch.from_numpy(data[i]).float().to(self.device))
                hyp = torch.from_numpy(data[1]).to(self.device).long()

                out, _ = self.classifier.forward(torch_data)

                fullout[:, step, :] = out
                fullhyp[:, step] = hyp
                self.current_seq_step += 1
        return fullhyp, fullout

    def predict_all_validation_subjects(self, dataloader, kfold,makeprints=True):
        validation_subjects = dataloader.validation_subjects_cv[kfold]
        cnn_acc = []; lstm_acc = []
        for i, subject in enumerate(validation_subjects):
            scores = self.evaluate_fullsubject(dataloader, subject, kfold)[2]
            cnn_acc.append(scores['cnn']['acc'])
            lstm_acc.append(scores['lstm']['acc'])
            if makeprints: print('For validation subject %d the CNN accuracy is %.3f and the LSTM accuracy is %.3f'
                                 %(subject, cnn_acc[-1], lstm_acc[-1]))
        return validation_subjects, cnn_acc, lstm_acc

    def load_classifier(self, kfold):
        if isinstance(self.classifier, (ensemble, expertnets_ensemble)):
            self.classifier.load_classifiers(kfold)
            print('Ensemble loaded')
        elif type(self.classifier) is EOG_1dcnn_cv or type(self.classifier) is EOG_double1dcnn_cv:
            self.classifier.load_epoch(self.classifier.savepath, self.epoch_to_load, kfold)
            print('CNN loaded')

    def plot_accpersubject(self, dataloader, meanacc=True): # invoke when fully trained (or last epoch of last fold is loaded)
        equallength = True
        minlength = len(self.epoch_time[0])
        for k in range(1, self.K):  # check that length is the same for all of the trainings
            if len(self.epoch_time[k]) < minlength:
                equallength = False
                minlength = len(self.epoch_time[k])

        if equallength is False: print("Not all trainings have the same length, so the minimum length will be used")


        if meanacc is True: # plot the evolution per fold
            colorlist = ['b', 'r', 'g', 'm', 'c']
            for ii, listacc in enumerate(self.mean_lstm_acc):
                plt.plot(listacc, label='Fold '+str(ii),color=colorlist[ii])
                plt.plot(self.mean_cnn_acc[ii],color=colorlist[ii], linestyle='dashed')
            plt.legend()
            plt.show()
        else:  # plot each individual subject, PER KFOLD (IF NOT, TOO MANY LINES)
            fig, axs = plt.subplots(nrows=1,ncols=dataloader.K,sharex='all', figsize=(20,10))
            for k in range(dataloader.K):
                nsubjects = len(dataloader.validation_subjects_cv[k])
                colorlist = self.get_cmap(nsubjects*2)
                subject_accs = np.zeros((nsubjects, minlength))
                subject_cnnaccs = np.zeros((nsubjects,minlength))
                for e, epochacc in enumerate(self.validation_subjects_lstm_acc[k]):  # select epoch
                    for s, subjectacc in enumerate(epochacc): # select each subject
                        subject_accs[s, e] = subjectacc
                        subject_cnnaccs[s, e] = self.validation_subjects_cnn_acc[k][e][s]
                for ii in range(nsubjects):
                    axs[k].plot(subject_accs[ii,:],label='Subject '+str(dataloader.validation_subjects_cv[k][ii]), color=colorlist(ii))
                    axs[k].plot(subject_cnnaccs[ii,:], color=colorlist(ii), linestyle='dashed')
                axs[k].set_ylim(0.45, 0.85)
                axs[k].legend()
            plt.show()

    def boxplot_accdistribution(self, dataloader, epoch):
        """plot the acc distribution for underlaying CNN and the LSTM on top
            at the desired epoch. Of course, training must be complete"""
        nsubjects = len(dataloader.hypnogram)
        subject_lstmaccs = np.zeros((nsubjects,))
        subject_cnnaccs = np.zeros((nsubjects,))
        for k in range(dataloader.K):
            val_subjects = dataloader.validation_subjects_cv[k]
            for s in range(len(val_subjects)):
                subject_lstmaccs[val_subjects[s]] = self.validation_subjects_lstm_acc[k][epoch][s]
                subject_cnnaccs[val_subjects[s]] = self.validation_subjects_cnn_acc[k][epoch][s]
        plt.boxplot([subject_cnnaccs, subject_lstmaccs],labels=['CNN','LSTM'])
        plt.title('Accuracy distribution for CNN and LSTM classifiers')
        plt.ylabel('Mean per-subject accuracy')
        plt.ylim(0.4,0.9)
        plt.show()

    @staticmethod
    def get_cmap(n, name='hsv'):
        """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name."""
        return plt.cm.get_cmap(name, n)
