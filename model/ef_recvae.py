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

        self.hidden_dim, self.latent_dim, self.input_dim = hidden_dim, latent_dim, input_dim
        # self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        # self.encoder_old.requires_grad_(False)

    def forward(self, x, z):          
            
        post_mu, post_logvar = self.encoder_old.forward_for_loss(x, 0)

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

        # Encoder Embedding & Bias
        self.enc_embeddings = nn.Embedding(input_dim, hidden_dim)
        self.enc_bias = nn.Parameter(torch.zeros((1, hidden_dim)))
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
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
        
        # Initilization
        for m in (self.enc_embeddings,):
            self.init_weights(m)
        torch.nn.init.normal_(self.enc_bias.data, 0, 0.001)

    def forward(self, x, dropout_prob):
        # x = F.normalize(x)
        # x = F.dropout(x, dropout_prob, training=self.training)
        # h1 = self.ln1(swish(self.fc1(x)))
        
        x = F.normalize(x, p=2, dim=1)
        x = F.dropout(x, dropout_prob, training=self.training)
        h = x @ self.enc_embeddings.weight + self.enc_bias
        h1 = self.ln1(swish(h))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        # h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        # h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h3), self.fc_logvar(h3)
    
    def forward_for_loss(self, x, dropout_prob):
        h = self.enc_embeddings(x) # batch_user * cur_max_len * dims
        h = F.dropout2d(h, dropout_prob, training=self.training)
        h = torch.sum(h, dim=1) / (x.shape[-1] ** 0.5 + 1e-8) + self.enc_bias
        h1 = self.ln1(swish(h))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        # h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        # h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h3), self.fc_logvar(h3)
    
    def init_weights(self, m):
        if isinstance(m, nn.Embedding):
            torch.nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias.data, 0, 0.001)


class EFRecVAE(nn.Module):
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
        # self.decoder = nn.Linear(self.latent_dim, self.n_items)
        
        # Decoder Embedding
        self.dec_embeddings = nn.Embedding(dataset.n_items, self.latent_dim)
        self.dec_bias = nn.Embedding(dataset.n_items, 1)
        
        for m in (self.dec_embeddings,):
            self.init_weights(m)
        torch.nn.init.normal_(self.dec_bias.weight.data, 0, 0.001)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def init_weights(self, m):
        if isinstance(m, nn.Embedding):
            torch.nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias.data, 0, 0.001)

    def forward(self, user_ratings, dropout_prob=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_prob)
        z = self.reparameterize(mu, logvar)
        x_pred = z @ self.dec_embeddings.weight.T + self.dec_bias.weight.T
        
        # if calculate_loss:
        #     if self.gamma:
        #         norm = user_ratings.sum(dim=-1)
        #         kl_weight = self.gamma * norm
        #     elif self.beta:
        #         kl_weight = self.beta

        #     mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
        #     kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
        #     negative_elbo = -(mll - kld)
            
        #     return (mll, kld), negative_elbo
            
        # else:
        return x_pred
        
    def calculate_loss(self, pos_items, sampler, dropout_prob=0.5):
        mu, logvar = self.encoder.forward_for_loss(pos_items, dropout_prob)
        z = self.reparameterize(mu, logvar)
        # h = self.decoder(z)
        
        with torch.no_grad():
            # t0 = time.time()
            # self.sampler.num_neg = pos_items.sum()
            pos_log_prob, neg_items, neg_log_prob = sampler(z, pos_items)
            # t1 = time.time()
        pos_items_emb = self.dec_embeddings(pos_items)
        neg_items_emb = self.dec_embeddings(neg_items)
        
        pos_items_bias = self.dec_bias(pos_items).squeeze(-1)
        neg_items_bias = self.dec_bias(neg_items).squeeze(-1)
        
        pos_logit = z @ pos_items_emb.permute(0, 2, 1) + pos_items_bias
        neg_logit = z @ neg_items_emb.permute(0, 2, 1) + neg_items_bias
        
        # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        if self.gamma:
            # norm = pos_items.sum(dim=-1)
            norm = pos_items.shape[-1]
            kl_weight = self.gamma * norm
        elif self.beta:
            kl_weight = self.beta
        
        KLD = (log_norm_pdf(z, mu, logvar) - self.prior(pos_items, z)).sum(dim=-1).mul(kl_weight).mean()
        
        ce_loss = self.loss_function(neg_logit, neg_log_prob, pos_logit, pos_log_prob)
        # x_pred = z @ self.dec_embeddings.weight.T + self.dec_bias.weight.T
        # ce_loss = -(F.log_softmax(x_pred, dim=-1)[0][pos_items[0]]).sum(dim=-1)
        
        loss = ce_loss + KLD
        
        return loss, neg_items.squeeze()
        # return loss, pos_items.squeeze()
    
    def loss_function(self, neg_logit, neg_log_prob, pos_logit, pos_log_prob):
        # idx_mtx = (pos_logit != 0).double()
        new_pos = pos_logit - pos_log_prob.detach()
        new_neg = neg_logit - neg_log_prob.detach()

        neg_log_sum_exp = torch.logsumexp(new_neg, dim=-1, keepdim=True)
        final = torch.logaddexp(new_pos, neg_log_sum_exp)
        
        # return torch.sum((- new_pos + final) * idx_mtx, dim=-1).mean()
        return torch.sum(-new_pos + final, dim=-1).mean()
    
    def construct_prior(self):
        self.prior.encoder_old = Encoder(self.hidden_dim, self.latent_dim, self.n_items).to(self.dec_bias.weight.device)
        self.prior.encoder_old.requires_grad_(False)
        # self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
        
    def remove_prior_encoder(self):
        del self.prior.encoder_old