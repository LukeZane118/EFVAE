import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    r"""Swish activation function:

    .. math::
        \text{Swish}(x) = \frac{x}{1 + \exp(-x)}
    """
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_prob):
        x = F.normalize(x)
        x = F.dropout(x, dropout_prob, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        # h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        # h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h3), self.fc_logvar(h3)


class RecVAE(nn.Module):
    r"""Collaborative Denoising Auto-Encoder (RecVAE) is a recommendation model
    for top-N recommendation with implicit feedback.

    We implement the model following the original author
    """

    def __init__(self, args, dataset):
        super().__init__()

        assert len(args.hidden_dim) == 2
        self.hidden_dim, self.latent_dim = args.hidden_dim
        self.n_items = dataset.n_items
        self.beta = args.beta
        self.mixture_weights = args.mixture_weights
        self.gamma = args.gamma
        
        self.encoder = Encoder(self.hidden_dim, self.latent_dim, self.n_items)
        self.prior = CompositePrior(self.hidden_dim, self.latent_dim, self.n_items, self.mixture_weights)
        self.decoder = nn.Linear(self.latent_dim, self.n_items)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, dropout_prob=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_prob=dropout_prob)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        
        if calculate_loss:
            if self.gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = self.gamma * norm
            elif self.beta:
                kl_weight = self.beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return (mll, kld), negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))