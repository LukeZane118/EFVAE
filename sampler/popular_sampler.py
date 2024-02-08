from .base import SamplerBase
import numpy as np
import torch

from utils import get_size


class PopularSampler(SamplerBase):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        item_count = torch.from_numpy(dataset.get_item_count()).to(self.device)
        if config.pop_count_mode == 0:
            pop_count = torch.log(item_count + 1)
        elif config.pop_count_mode == 1:
            pop_count = torch.log(item_count) + 1e-08
        elif config.pop_count_mode == 2:
            pop_count = item_count ** 0.75
        
        # pop_count = torch.cat([torch.zeros(1, device=self.device), pop_count])
        self.pop_prob = pop_count / pop_count.sum()
        # self.table = torch.cumsum(self.pop_prob, -1)
        # self.pop_prob[0] = torch.ones(1, device=self.device)
        self.table = torch.cumsum(self.pop_prob, dim=0)
    
    @torch.no_grad()
    def forward(self, query, pos_items=None, item_embs=None, temp=1.):
        num_queries = np.prod(query.shape[0])
        seeds = torch.rand(num_queries, self.num_neg, device=self.device)
        neg_items = torch.searchsorted(self.table, seeds)
        # neg_items = np.random.choice(range(0, self.num_item), size=num_queries * self.num_neg, replace=True, p=self.pop_prob.cpu())
        # neg_items = torch.from_numpy(neg_items).view(*query.shape[:-1], -1).to(self.device)
        neg_log_probs = torch.log(self.pop_prob[neg_items])
        pos_log_probs = None
        if pos_items is not None:
            pos_log_probs = torch.log(self.pop_prob[pos_items])
        return pos_log_probs, neg_items, neg_log_probs
    
    def get_communication_cost(self):
        return 0

