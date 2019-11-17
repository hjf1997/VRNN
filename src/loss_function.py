# implemented by p0werHu
# 11/17/2019

import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
import torch


def loss(package, x):

    prior_means, prior_var, decoder_means, decoder_var, x_decoded = package
    loss = 0.
    for i in range(x.shape[1]):
        # Kl loss
        norm_dis1 = Norm.Normal(prior_means[i], prior_var[i])
        norm_dis2 = Norm.Normal(decoder_means[i], decoder_var[i])
        kl_loss = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))

        # reconstruction loss
        xent_loss = torch.mean(F.binary_cross_entropy(x_decoded[i], x[:, i, :], reduction='none'))
        loss += xent_loss + kl_loss

    return loss