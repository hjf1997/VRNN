# implemented by p0werHu
# 11/15/2019

import torch
import torch.nn as nn
import numpy as np

class VRNN(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):
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
            nn.Linear(x_dim, h_dim),
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
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # decoder: input phi(z), h
        self.decoder_fea = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.decoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # encoder: input: phi(x), h
        self.decoder_fea = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        # VRE regard mean values sampled from z as the output
        self.decoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # using the recurrence equation to update its hidden state
        self.rnn = nn.GRU(h_dim*2, h_dim, batch_first=True)

    def forward(self, *input):
        pass

