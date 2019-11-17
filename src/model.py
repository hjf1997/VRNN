# implemented by p0werHu
# 11/15/2019

import torch
import torch.nn as nn
import numpy as np
import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F

class VRNN(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        # feature extractors of x and z
        # paper: We found that these feature extractors are crucial for learnting complex sequences
        # paper: 'all of phi_t have four hidden layers using rectificed linear units ReLu'
        self.x_fea = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.z_fea = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        # prior: input h output mu, sigma
        self.prior_fea = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.prior_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # decoder: input phi(z), h
        self.decoder_fea = nn.Sequential(
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU()
        )
        self.decoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )

        # encoder: input: phi(x), h
        self.encoder_fea = nn.Sequential(
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU()
        )
        # VRE regard mean values sampled from z as the output
        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # using the recurrence equation to update its hidden state
        self.rnn = nn.GRUCell(h_dim*2, h_dim)

    def forward(self, x):
        """

        :param x: shape of [frame, batch, features]
        :return:
        """
        h = torch.zeros([x.shape[0], self.h_dim], device=x.device)
        prior_means_all = []
        prior_var_all = []
        encoder_means_all = []
        encoder_var_all = []
        decoder_means_all = []
        for time in range(x.shape[1]):
            # feature extractor:
            phi_x = self.x_fea(x[:, time, :])

            # prior
            prior_fea_ = self.prior_fea(h)
            prior_means_ = self.prior_mean(prior_fea_)
            prior_var_ = self.prior_var(prior_fea_)

            # encoder
            encoder_fea_ = self.encoder_fea(torch.cat([phi_x, h], dim=1))
            encoder_means_ = self.encoder_mean(encoder_fea_)
            encoder_var_ = self.encoder_var(encoder_fea_)

            # decoder
            z_sampled = self.reparametrizing(encoder_means_, encoder_var_)
            phi_z = self.z_fea(z_sampled)
            decoder_fea_ = self.decoder_fea(torch.cat([phi_z, h], dim=1))
            decoder_means_ = self.decoder_mean(decoder_fea_)
            decoder_var_ = self.decoder_var(decoder_fea_)

            prior_means_all.append(prior_means_)
            prior_var_all.append(prior_var_)
            encoder_means_all.append(encoder_means_)
            encoder_var_all.append(encoder_var_)
            decoder_means_all.append(decoder_means_)
            # rnn
            h = self.rnn(torch.cat([phi_x, phi_z], dim=1), h)

        return [prior_means_all, prior_var_all, encoder_means_all, encoder_var_all, decoder_means_all]

    def sampling(self, seq_len, device):

        sample = torch.zeros(seq_len, self.x_dim, device=device)
        h = torch.zeros(1, self.h_dim, device=device)

        for t in range(seq_len):
            # prior
            prior_fea_ = self.prior_fea(h)
            prior_means_ = self.prior_mean(prior_fea_)
            prior_var_ = self.prior_var(prior_fea_)

            # decoder
            z_t = self.reparametrizing(prior_means_, prior_var_)
            phi_z_t = self.z_fea(z_t)
            decoder_fea_ = self.decoder_fea(torch.cat([phi_z_t, h], dim=1))
            decoder_means_ = self.decoder_mean(decoder_fea_)

            phi_x_t = self.x_fea(decoder_means_)
            # rnn
            h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)

            sample[t] = decoder_means_.detach()

        return sample

    def reparametrizing(self, *args):
        z_mean, z_log_var = args
        epsilon = torch.rand_like(z_mean, device=z_mean.device)
        return z_mean + z_log_var * epsilon





