from cmath import log
from .base import SamplerBase
from operator import imod, neg
import numpy as np
from numpy.core.numeric import indices
import scipy.sparse as sps
from sklearn import cluster
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch._C import device, dtype


class UniformSampler(SamplerBase):
    """
        Uniformly Sample negative items for each query. 
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.log_prob = -log(self.num_item)
    
    @torch.no_grad()
    def forward(self, query, pos_items=None, item_embs=None, temp=1.):
        num_queries = np.prod(query.shape[0])
        neg_items = torch.randint(0, self.num_item, size=(num_queries, self.num_neg), device=self.device)
        # neg_items = neg_items.view(*query.shape[:-1], -1)
        # neg_log_probs = -torch.log(self.num_item * torch.ones_like(neg_items, dtype=torch.float))
        neg_log_probs = torch.full_like(neg_items, self.log_prob)
        pos_log_probs = None
        if pos_items is not None:
            # pos_log_probs = -torch.log(self.num_item * torch.ones_like(pos_items, dtype=torch.float))
            # pos_log_probs = torch.full_like(pos_items, self.log_prob)
            pos_log_probs = self.compute_item_p(query, pos_items)
        return pos_log_probs, neg_items, neg_log_probs

    @torch.no_grad()
    def compute_item_p(self, query, pos_items):
        return torch.full_like(pos_items, self.log_prob) 
    
    def get_communication_cost(self):
        return 0
    