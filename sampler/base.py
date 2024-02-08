from operator import imod, neg
from turtle import forward
import numpy as np
from numpy.core.numeric import indices
import scipy.sparse as sps
import torch
import torch.nn as nn
from torch._C import device, dtype


class SamplerBase(nn.Module):
    """
        Abstract class to define sampler. 
    """
    def __init__(self, config, dataset):
        super().__init__()
        self.num_item = dataset.n_items
        self.num_neg = config.num_sample
        self.device = self.device = f"cuda:{config.gpu_id}" if torch.cuda.is_available() and config.gpu_id >= 0 else "cpu"

    @torch.no_grad()
    def forward(self, query, pos_items=None, item_embs=None, padding=0):
        """
        Input
            query: torch.tensor
                Sequential models:
                query: (B,L,D), pos_items: (B,L)
                Normal models:
                query: (B,D), pos_items: (B,L)
        Output
            pos_prob(None if no pos_items), neg_items, neg_prob
            pos_items.shape == pos_prob.shape
            neg_items.shape == neg_prob.shape
            Sequential models:
            neg_items: (B,L,N)
            Normal
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample_item(self, k01, p01):
        """Sampler negtive item based on codebook."""
        raise NotImplementedError
    
    @torch.no_grad()
    def compute_item_p(self, query, pos_items):
        """Compute probability of being sampled for positive items."""
        raise NotImplementedError
    
    def get_communication_cost(self):
        raise NotImplementedError
