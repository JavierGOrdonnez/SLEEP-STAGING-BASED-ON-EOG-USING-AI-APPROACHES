"""Variational Recurrent Neural Network,  which uses the hidden state of an RNN as prior to the latent space distribution
of a Variational Autoencoder. Based on the article "A Recurrent Latent Variable Model for Sequential Data" by Chung et al. 2017
and the existing PyTorch implementation in https://github.com/emited/VariationalRecurrentNeuralNetwork"""

import math
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as pltr

class VRNN(nn.Module):
    def __init__(self, ensemble, h_dim, z_dim, n_layers, bias=False,
                 lr=1e-4, optimizer='Adam', savepath='D:/GitHub/TFM/07_VRNN/saves/ontopensemble/',
                 beta=1, clip=10):
        super().__init__()

        self.x_dim = ensemble.outputsize
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.ensemble = ensemble
        self.lr = lr
        self.beta = beta  # controls tradeoff KLdiv - MSE
        self.clip = clip
        self.optimizer = optimizer.lower()

        if savepath[-1] is '/': savepath = savepath[:-1]  # remove the final /. It will be appended later.
        self.savepath = savepath
        if os.path.isdir(savepath) is False: os.mkdir(savepath)

        self.KLDloss_during_training = [[] for _ in range(self.ensemble.K)]
        self.MSEloss_during_training = [[] for _ in range(self.ensemble.K)]
        self.KLDloss_valid_during_training = [[] for _ in range(self.ensemble.K)]
        self.MSEloss_valid_during_training = [[] for _ in range(self.ensemble.K)]
        self.KLDloss_per_batch = [[] for _ in range(self.ensemble.K)]
        self.MSEloss_per_batch = [[] for _ in range(self.ensemble.K)]

        self.classification_accuracy = [[] for _ in range(self.ensemble.K)]

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, self.x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(h_dim, self.x_dim)

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
        # to use a LSTM, remember that it unpacks _ , (h,c)

        self.latent_space_classifier = nn.Sequential(
            nn.Linear(z_dim, 5),
            nn.LogSoftmax(dim=1)
        )
        self.classification_criterion = nn.NLLLoss()

        self.device = self.ensemble.device
        self.to(self.device)

        params_list = [{'params': self.rnn.parameters()},
                       {'params': self.phi_x.parameters()},
                       {'params': self.phi_z.parameters()},
                       {'params': self.enc.parameters()},
                       {'params': self.enc_mean.parameters()},
                       {'params': self.enc_std.parameters()},
                       {'params': self.prior.parameters()},
                       {'params': self.prior_mean.parameters()},
                       {'params': self.prior_std.parameters()},
                       {'params': self.dec.parameters()},
                       {'params': self.dec_mean.parameters()},
                       {'params': self.dec_std.parameters()}]

        if self.optimizer == 'adam':
            self.optim = optim.Adam(params_list, self.lr)
        elif self.optimizer == 'sgd':
            self.optim = optim.SGD(params_list, self.lr)
        elif self.optimizer == 'momentum':
            self.optim = optim.SGD(params_list, self.lr, momentum=0.9)
        elif self.optimizer == 'adagrad':
            self.optim = optim.Adagrad(params_list, self.lr)
        elif self.optimizer == 'adadelta':
            self.optim = optim.Adadelta(params_list)
        elif self.optimizer == 'rmsprop':
            self.optim = optim.RMSprop(params_list, self.lr)
        else:
            assert 0, 'Requested optimizer not recognized: ' + str(self.optimizer)

        self.optim_classifier = optim.SGD(self.latent_space_classifier.parameters(),lr=1e-4)

    def forward(self, features):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        mse_loss = 0  # now it is MSE, just I didnt change the name

        # features = [seq_len x batch_size x num_features]
        self.z = Variable(torch.zeros((features.size(0), features.size(1), self.z_dim))).to(self.device)
        # to later be able to access the latent representation
        h = Variable(torch.zeros(self.n_layers, features.size(1), self.h_dim)).to(self.device)
        for t in range(features.size(0)):  # x.size0 is seq_len; x.size1 is batch_size

            phi_x_t = self.phi_x(features[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            self.z[t, :, :] = z_t.detach()# .cpu().numpy()
            # to later be able to access the latent representation
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            # I dont know why it is applying NLLLoss (isnt it for classification, after softmax? )
            mse_loss += self._my_reconstruction_loss(dec_mean_t, features[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, mse_loss, \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std)

    def trainloop(self, dataloader, kfold, epochs, batch_size=25, seq_len=100, shuffle=True, shuffling_seed='fixed', epochs_wo_kl=0):
        self.current_fold = kfold
        self.ensemble.load_classifiers(kfold)
        self.batch_size_evaluation = batch_size

        self.current_epoch = 0
        for e in range(epochs):
            t_start_epoch = time.time()
            self.current_epoch += 1
            self.current_seq_step = 0
            batch_idx = 0
            epoch_loss = 0.
            mean_kld_loss, mean_mse_loss = 0., 0.
            self.train()
            if shuffling_seed is 'fixed':
                seed = e  # allow to always use a different batch division each epoch, but always the same
            else:
                seed = np.random.randint(100)  # or let it be random

            generator = dataloader.batch_generator(kfold, batch_size, shuffle=shuffle, shuffling_seed=seed)

            # train_loader = d.batch_generator(batch_size=batch_size, shuffle=True, shuffling_seed=epoch)
            N = len(dataloader.hypnogram_cv_train[kfold]) // batch_size
            assert N > seq_len, 'Seq_len ' + str(seq_len) + ' * batch_size ' + str(batch_size) + ' is greater than the' \
                                                                                                 ' number of available samples ' + str(
                len(dataloader.hypnogram_cv_train[kfold])) + \
                                '. Please consider reducing the parameters so it is possible to complete at least ' \
                                'one batch per epoch.'
            fullout = torch.zeros((seq_len, batch_size, self.ensemble.outputsize)).to(self.device)
            fullhyp = torch.zeros((seq_len, batch_size)).to(self.device)
            # get the full set of features [seq_len x batch_size x num_features]
            self.current_seq_step = 0
            self.ensemble.eval()
            for data in generator:
                # data is a bunch of different features in dataloader_v5
                # data = (eog, h, mse_coeffs, psd_coeffs, psd_highres_coeffs, spect_values, stats_values, AR_coefs_values)

                self.current_seq_step += 1
                with torch.no_grad():
                    torch_data = []
                    for i in range(len(data)):
                        torch_data.append(torch.from_numpy(data[i]).float().to(self.device))
                        # convert to Torch each element of the tuple
                    hyp = torch.from_numpy(data[1]).to(self.device).long()
                    features, _ = self.ensemble.forward(torch_data)
                    # necessary shape: [seq_len, batch_size, x_dim]
                    fullout[self.current_seq_step - 1, :, :] = features
                    fullhyp[self.current_seq_step - 1, :] = hyp

                if self.current_seq_step == seq_len:
                    self.optim.zero_grad()
                    self.train()
                    self.ensemble.eval()
                    kld_loss, mse_loss, _, _ = self.forward(fullout)
                    if self.current_epoch <= epochs_wo_kl:
                        loss = mse_loss
                    else:
                        loss = mse_loss + self.beta * kld_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                    self.optim.step()

                    # also optimize the 2 linear layers that classify from z
                    self.optim_classifier.zero_grad()
                    classification = self.latent_space_classifier(self.z).reshape(-1,5)
                    loss2 = self.classification_criterion(classification, fullhyp.reshape(-1,).long())
                    loss2.backward()
                    self.optim_classifier.step()
                    _, topclass = classification.topk(1, dim=-1)
                    self.classification_accuracy[kfold].append(
                        accuracy_score( fullhyp.reshape(-1,).long().detach().cpu().numpy(),
                                        topclass.detach().cpu().numpy() ))

                    mean_kld_loss += kld_loss.item()
                    mean_mse_loss += mse_loss.item()
                    epoch_loss += loss.item()
                    self.KLDloss_per_batch[kfold].append(kld_loss.item() / (batch_size * seq_len) )
                    self.MSEloss_per_batch[kfold].append(mse_loss.item() / (batch_size * seq_len) )

                    fullout = torch.zeros((seq_len, batch_size, self.ensemble.outputsize)).to(self.device)
                    fullhyp = torch.zeros((seq_len, batch_size)).to(self.device)
                    self.current_seq_step = 0  # reinitialize this counter, but not the total
                    batch_idx += 1
                    print('Train Epoch: {} [batch {}]\tKLD Loss: {:.3f} \tMSE Loss: {:.3f} \tAccuracy: {:.3f}'.format(
                        e, batch_idx, self.KLDloss_per_batch[kfold][-1], self.MSEloss_per_batch[kfold][-1],
                        self.classification_accuracy[kfold][-1]))

            if batch_idx == 0:
                # when a batch can not be completed in the seq dimension, it is not used
                # raise an alert when no batch at all was completed (and therefore no training)
                print("The mechanism above did not work. Revise it.")
                assert 0, "Seq_len * batch_size is greater that the number of available samples, " \
                          "so it was not possible to train"

            self.KLDloss_during_training[kfold].append(mean_kld_loss / (batch_idx*batch_size * seq_len))
            self.MSEloss_during_training[kfold].append(mean_mse_loss / (batch_idx*batch_size * seq_len))
            print('====> Epoch: {} Average losses: KLD Loss = {:.3f}, MSE Loss = {:.3f}. '
                  'Time taken: {:.2f} seconds.'.format(e, self.KLDloss_during_training[kfold][-1],
                    self.MSEloss_during_training[kfold][-1], (time.time() - t_start_epoch)))
            self.evaluate(dataloader, kfold, seq_len)
            self.save_during_training()

    def save_during_training(self):
        # save weights (state_dict)
        savename = '{0}/epoch{1:d}_cv{2:d}.pth'.format(self.savepath, self.current_epoch, self.current_fold)
        torch.save(self.state_dict(), savename)

    def evaluate(self, dataloader, kfold, seq_len=100):
        if self.current_fold != kfold:
            self.current_fold = kfold
            self.ensemble.load_classifiers(kfold)

        with torch.no_grad():
            mean_kld_loss, mean_mse_loss = 0, 0
            self.eval()
            self.ensemble.eval()
            if self.batch_size_evaluation * seq_len > len(dataloader.hypnogram_cv_valid[kfold]):
                print('Seq_len ' + str(seq_len) + ' * batch_size ' + str(self.batch_size_evaluation)
                      + ' was greater than the available samples for validation ' + str(
                    len(dataloader.hypnogram_cv_valid[kfold]))
                      + '. Trying again with half the batch size.')
                if self.batch_size_evaluation >= 2:
                    self.batch_size_evaluation = self.batch_size_evaluation // 2
                    self.evaluate(dataloader, kfold, seq_len)
                    return
                else:
                    assert 0, "Seq_len " + str(seq_len) + " is too large. Not possible to do any evaluation."
            # I can set this so that we do everything in a single batch (faster and simpler)
            generator = dataloader.batch_generator(kfold, self.batch_size_evaluation, validation_set=True,
                                                   shuffle=False)
            fullout = torch.zeros((seq_len, self.batch_size_evaluation, self.ensemble.outputsize)).to(self.device)
            self.ensemble.eval()
            self.current_seq_step = 0
            batch_idx = 0
            for data in generator:
                # data is a bunch of different features in dataloader_v5
                # data = (eog, h, mse_coeffs, psd_coeffs, psd_highres_coeffs, spect_values, stats_values, AR_coefs_values)
                torch_data = []
                for i in range(len(data)):
                    torch_data.append(torch.from_numpy(data[i]).float().to(self.device))
                    # convert to Torch each element of the tuple
                features, _ = self.ensemble.forward(torch_data)
                # necessary shape: [seq_len, batch_size, x_dim]
                fullout[self.current_seq_step, :, :] = features
                self.current_seq_step += 1

                if self.current_seq_step == seq_len:
                    kld_loss, mse_loss, _, _ = self.forward(fullout)
                    mean_kld_loss += kld_loss.item()
                    mean_mse_loss += mse_loss.item()
                    self.current_seq_step = 0  # reinitialize this counter, but not the total
                    batch_idx += 1
                    # print('Validation [batch {}]\t KLD Loss: {:.6f} \t MSE Loss: {:.6f}'.format(
                    #         batch_idx, kld_loss.item(), mse_loss.item()))

            if batch_idx == 0:
                # when a batch can not be completed in the seq dimension, it is not used
                # raise an alert when no batch at all was completed (and therefore no training)
                assert 0, "The batch size reduction mechanism has not worked"

            self.KLDloss_valid_during_training[kfold].append(mean_kld_loss / (batch_idx * self.batch_size_evaluation * seq_len))
            self.MSEloss_valid_during_training[kfold].append(mean_mse_loss / (batch_idx * self.batch_size_evaluation * seq_len))

            print('====> Validation set loss: KLD Loss = {:.4f}, MSE Loss = {:.4f} \n'.format(
                self.KLDloss_valid_during_training[kfold][-1],
                self.MSEloss_valid_during_training[kfold][-1]))

    def sample(self, seq_len):
        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    # def _nll_bernoulli(self, theta, x):
    #     return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))
    #
    #
    # def _nll_gauss(self, mean, std, x):
    #     pass

    def _my_reconstruction_loss(self, dec_mean_t, x):
        loss_function = nn.MSELoss()
        # nn.L1Loss
        loss = loss_function(dec_mean_t, x)
        return loss

    def plot_training_information(self):  # invoke when fully trained (or last epoch loaded)
        equallength = True
        minlength = len(self.KLDloss_during_training[0])
        for k in range(1, self.K):  # check that length is the same for all of the trainings
            if len(self.KLDloss_during_training[k]) < minlength:
                equallength = False
                minlength = len(self.KLDloss_during_training[k])

        if equallength is False: print("Not all trainings have the same length, so the minimum length will be used")

        L = minlength
        full_train_KLDloss = np.zeros((self.K, L))
        full_train_MSEloss = np.zeros((self.K, L))
        full_valid_KLDloss = np.zeros((self.K, L))
        full_valid_MSEloss = np.zeros((self.K, L))


        for k in range(self.K):
            full_train_KLDloss[k, :] = self.KLDloss_during_training[k][:L]
            full_train_MSEloss[k, :] = self.MSEloss_during_training[k][:L]
            full_valid_KLDloss[k, :] = self.KLDloss_valid_during_training[k][:L]
            full_valid_MSEloss[k, :] = self.MSEloss_valid_during_training[k][:L]

        mean_train_kld = np.mean(full_train_KLDloss, axis=0)
        mean_train_mse = np.mean(full_train_MSEloss, axis=0)
        mean_valid_kld = np.mean(full_valid_KLDloss, axis=0)
        mean_valid_mse = np.mean(full_valid_MSEloss, axis=0)
        best_epoch = int(np.argmin(mean_valid_kld + mean_valid_mse))

        std_train_kld = np.std(full_train_KLDloss, axis=0)
        std_train_mse = np.std(full_train_MSEloss, axis=0)
        std_valid_kld = np.std(full_valid_KLDloss, axis=0)
        std_valid_mse = np.std(full_valid_MSEloss, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        x = np.arange(len(mean_train_kld)) + 1
        ax1.plot(x, mean_train_kld, 'r.-', label='Training KL divergence ($\pm$std)')
        ax1.fill_between(x, mean_train_kld - std_train_kld, mean_train_kld + std_train_kld, facecolor="r", alpha=0.5)
        ax1.plot(x, mean_valid_kld, 'b--', label='Validation KL divergence ($\pm$std)')
        ax1.fill_between(x, mean_valid_kld - std_valid_kld, mean_valid_kld + std_valid_kld, facecolor="b", alpha=0.5)

        ax2.plot(x, mean_train_mse, 'r.-', label='Training MSE ($\pm$std)')
        ax2.fill_between(x, mean_train_mse - std_train_mse, mean_train_mse + std_train_mse, facecolor="r", alpha=0.5)
        ax2.plot(x, mean_valid_mse, 'b--', label='Validation MSE ($\pm$std)')
        ax2.fill_between(x, mean_valid_mse - std_valid_mse, mean_valid_mse + std_valid_mse, facecolor="b", alpha=0.5)

        ax1.legend()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('KL divergence')
        ax1.set_title('KL divergence during training')

        ax2.set_xlabel('Epochs')
        ax2.set_ylabel("MSE")
        ax2.set_title('MSE during training')

        plt.savefig(self.savepath + '/fulltraining.png')
        plt.show()

        # print the best epochs in terms of loss and accuracy
        print('Epoch %d has the lowest validation loss (KL: %.3f + MSE: %.3f)' %(best_epoch, mean_train_kld[best_epoch],
                                                                                 mean_train_mse[best_epoch]))
        return mean_valid_kld, mean_valid_mse

    def plot_training_information_onefold(self, k=0):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10, 10))
        x = np.arange(len(self.KLDloss_during_training[k]))
        ax1.semilogy(x, self.KLDloss_during_training[k], 'r.-', label='Training KL divergence')
        ax1.semilogy(x, self.KLDloss_valid_during_training[k], 'b--', label='Validation KL divergence')

        ax2.semilogy(x, self.MSEloss_during_training[k], 'r.-', label='Training MSE')
        ax2.semilogy(x, self.MSEloss_valid_during_training[k], 'b--', label='Validation MSE')

        ax1.legend()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('KL divergence')
        ax1.set_title('KL divergence during training')

        ax2.legend()
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel("MSE")
        ax2.set_title('MSE during training')

        ax3.semilogy(self.KLDloss_per_batch[k], 'r.-')
        ax3.title('Per batch training KLD')
        ax3.set_xlabel('Batches')
        ax3.set_ylabel('KL divergence')

        ax4.semilogy(self.MSEloss_per_batch[k], 'r.-')
        ax4.title('Per batch training MSE')
        ax4.set_xlabel('Batches')
        ax4.set_ylabel("MSE")


        plt.show()

    def visualize_latent_representation(self, dataloader, subject, normalize_z = True):
        data = (dataloader.pEOG[subject], dataloader.hypnogram[subject], np.zeros(dataloader.PSD[subject].shape),#dataloader.MSE[subject],
                dataloader.PSD[subject], dataloader.PSD_highres[subject], dataloader.spect[subject],
                dataloader.stats[subject], dataloader.AR_coefs[subject])
        torch_data = []
        for i in range(len(data)):
            torch_data.append(torch.from_numpy(data[i]).float().to(self.device))
        features, _ = self.ensemble.forward(torch_data)
        features = features.unsqueeze(1)  # batch_size = 1
        _ = self.forward(features.float())
        z = self.z.squeeze(1)
        z = z.detach().cpu().numpy() # [seq_len, z_dim]
        from sklearn.manifold import MDS
        from sklearn.preprocessing import MinMaxScaler
        if normalize_z:
            scaler = MinMaxScaler()
            z = scaler.fit_transform(z)
        mds = MDS(2, random_state=0)
        z_2d = mds.fit_transform(z)

        colors = ['black', 'green', 'blue', 'orange', 'red']
        labels = ['W', 'N1', 'N2', 'N3', 'REM']
        markers = ['x', '*', 's', '^', 'o']
        plt.rc('font', size=14)
        plt.figure(figsize=(7,7))
        for i in np.unique(dataloader.hypnogram[subject]):
            subset = z_2d[dataloader.hypnogram[subject] == i]

            x = [row[0] for row in subset]
            y = [row[1] for row in subset]
            plt.scatter(x, y, c=colors[i], marker=markers[i], label=labels[i])
        plt.legend()
        plt.show()


def reshape_EOG_data(data):
    Fs = 50
    # data = [batch_size, 2, 1500]
    # it should be [seq_len, batch_size, x_dim]
    # 1: swap batch size to pos 1
    data = torch.from_numpy(data)
    data = Variable(data.transpose(0, 1))
    # 2: retain just left EOG
    data = data[0]  # now data = [batchsize, x_dim]
    # 3: rearrange in 1 second segments (Fs=50 samples) as seq len
    # so that we finally have [30, batchsize, Fs]
    data = data.reshape(data.shape[0], -1, Fs)
    data = data.transpose(0, 1)
    # data = (data - data.min().item()) / (data.max().item() - data.min().item())
    return data
