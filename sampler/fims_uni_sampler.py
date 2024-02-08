from .base import SamplerBase
import torch
from fast_pytorch_kmeans import KMeans
# from torch_kmeans import KMeans

from sampler_utils import *
from utils import get_size


class FIMSUniSampler(SamplerBase):
    """
        FIMS-Uni Sampler.
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.num_cluster = config.num_cluster
        self.num_code = self.num_cluster * self.num_cluster
    
    # def forward2(self, query, pos_items=None, padding=0):
    #     # assert padding == 0
    #     q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
    #     r1 = q1 @ self.c1.T
    #     r1s = torch.softmax(r1, dim=-1) # num_q x K1
    #     r0 = q0 @ self.c0.T
    #     r0s = torch.softmax(r0, dim=-1) # num_q x K0
    #     s0 = (r1s @ self.wkk.T) * r0s # num_q x K0 | wkk: K0 x K1
    #     k0 = torch.multinomial(s0, self.num_neg, replacement=True) # num_q x neg
    #     p0 = torch.gather(r0, -1, k0)     # num_q * neg
            
    #     subwkk = self.wkk[k0, :]          # num_q x neg x K1
    #     s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
    #     k1 = torch.multinomial(s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1]) # num_q x neg
    #     p1 = torch.gather(r1, -1, k1) # num_q x neg
    #     k01 = k0 * self.num_cluster + k1  # num_q x neg
    #     p01 = p0 + p1
    #     neg_items, neg_prob = self.sample_item(k01, p01)
    #     if pos_items is not None:
    #         pos_prop = self.compute_item_p(query, pos_items)
    #         return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
    #     return None, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    @torch.no_grad()
    def forward(self, query, pos_items=None, item_embs=None, temp=1.):
        # query: B x D | B x T x D
        # pos_items: B x P x D
        # cates: I x T
        # T = 1 if len(query.shape) == 2 else query.shape[1]
        q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)  # B x D/2
        logit0 = q0 @ self.cb0.T  # B x C0
        logit1 = q1 @ self.cb1.T  # B x C1
        logit = (logit0.unsqueeze(1).permute(0, 2, 1).repeat(1, 1, self.num_cluster) + logit1.unsqueeze(1).repeat(1, self.num_cluster, 1)).contiguous()
        logit = logit.view(len(query), -1)  # B x (C0 * C1)
        
        sample_pro = torch.softmax(logit, dim=-1)  # B x (C0 * C1)
        # if len(query.size()) > 2 or cates is not None:
            # probs = torch.exp(logit)  # B x (C0 * C1)
        
        # sample_pro = sample_pro.view(len(query), -1, self.num_code)  # B x T x (C0 * C1)
        sample_pro = self.cd2weights * sample_pro  # B x (C0 * C1)
        sample_code = torch.multinomial(sample_pro, self.num_neg, replacement=True)  # B x N
        
        # sample_code_repeat = sample_code.unsqueeze(1).repeat(1, T, 1)  # B x T x N
        sample_logit = torch.gather(logit, -1, sample_code)  # B x N
        
        # if self.bias is not None:
        #     sample_cates = torch.gather(self.bias.T.unsqueeze(0).repeat(len(sample_code_repeat), 1, 1), -1, sample_code_repeat)
        #     # sample_logit = (sample_cates * torch.exp(sample_logit)).sum(1).log()  # B x N
        #     sample_logit = torch.logsumexp(torch.log(sample_cates) + sample_logit, dim=1)
        # else:
        #     sample_logit = sample_logit.squeeze(1)
        
        if self.bias is None:
            neg_items, neg_log_probs = self.sample_item(sample_code, sample_logit)
        else:
            neg_items, neg_log_probs = self.sample_item_bias(sample_code, sample_logit)
        
        pos_log_probs = None
        if pos_items is not None:
            pos_log_probs = self.compute_item_p(query, pos_items)
            
        return pos_log_probs, neg_items.view(*query.shape[:-1], -1), neg_log_probs.view(*query.shape[:-1], -1)
    
    @torch.no_grad()
    def sample_item(self, cd01, log_prob):
        item_cnt = self.indptr[cd01 + 1] - self.indptr[cd01] # num_q x neg, the number of items
        item_idx = torch.floor(item_cnt * torch.rand_like(item_cnt, dtype=torch.float32, device=self.device)).long() # num_q x neg
        neg_items = self.indices[item_idx + self.indptr[cd01]]
        neg_log_probs = log_prob
        return neg_items, neg_log_probs
    
    @torch.no_grad()
    def sample_item_bias(self, cd01, log_prob):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[cd01]  # B x N
        last = self.indptr[cd01 + 1] - 1  # B x N
        count = last - start + 1
        maxlen = count.max()
        # print(maxlen)
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen, device=self.device).reshape(1, 1, maxlen)  # B x N x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))  # B x N x maxlen
        item_idx = torch.searchsorted(self.cumBias[fullrange], torch.rand_like(start, dtype=torch.float32, device=self.device).unsqueeze(-1)).squeeze(-1)  # B x N
        item_idx = torch.minimum(item_idx + start, last)  # B x N
        neg_items = self.indices[item_idx]
        # neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.bias[neg_items] if self.bias is not None else 0
        return neg_items, log_prob + neg_probs
    
    @torch.no_grad()
    def compute_item_p(self, query, pos_items):
        # query: B x D, pos_items: B x L
        cd0 = self.i2cd0[pos_items]  # B x L
        cd1 = self.i2cd1[pos_items]  # B x L
        c0 = self.cb0[cd0, :]  # B x L x D/2
        c1 = self.cb1[cd1, :]  # B x L x D/2
        q0, q1 = query.chunk(2, dim=-1)  # B x D/2
        if query.dim() == pos_items.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1) # B x L
        else:
            r = torch.sum(c0 * q0, dim=-1) + torch.sum(c1 * q1, dim=-1) # B x L
        p_b = self.bias[pos_items] if self.bias is not None else 0
        return r + p_b

    @torch.no_grad()
    def update(self, item_embs, bias=None):
        # items_embs: I x D
        # cates: I x T
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        
        # self.c0, cd0, cd0m, _ = kmeans(embs1, self.num_cluster)
        # self.c1, cd1, cd1m, _ = kmeans(embs2, self.num_cluster)
        
        # cd0, self.c0 = kmeans_pytorch.kmeans(X=embs1, num_clusters=self.num_cluster, distance='euclidean', device=self.device)
        # cd1, self.c1 = kmeans_pytorch.kmeans(X=embs1, num_clusters=self.num_cluster, distance='euclidean', device=self.device)
        
        kmeans = KMeans(n_clusters=self.num_cluster, max_iter=300, tol=1e-6, verbose=0, mode="euclidean", init_method="kmeans++")
        self.i2cd0 = kmeans.fit_predict(embs1)
        self.cb0 = kmeans.centroids
        # kmeans = KMeans(n_clusters=self.num_cluster, init_method="k-means++", num_init=4, max_iter=300, verbose=False, tol=1e-6)
        # kmeans = kmeans.fit(embs1.unsqueeze(0))
        # self.i2cd0 = kmeans._result.labels.squeeze(0)
        # self.cb0 = kmeans._result.centers.squeeze(0)
        
        kmeans = KMeans(n_clusters=self.num_cluster, max_iter=300, tol=1e-6, verbose=0, mode="euclidean", init_method="kmeans++")
        self.i2cd1 = kmeans.fit_predict(embs2)
        self.cb1 = kmeans.centroids
        # kmeans = KMeans(n_clusters=self.num_cluster, init_method="k-means++", num_init=4, max_iter=300, verbose=False, tol=1e-6)
        # kmeans = kmeans.fit(embs2.unsqueeze(0))
        # self.i2cd1 = kmeans._result.labels.squeeze(0)
        # self.cb1 = kmeans._result.centers.squeeze(0)

        # self.c0_ = torch.cat([torch.zeros(1, self.c0.size(1), device=self.device), self.c0], dim=0) ## for retreival probability, considering padding
        # self.c1_ = torch.cat([torch.zeros(1, self.c1.size(1), device=self.device), self.c1], dim=0) ## for retreival probability, considering padding

        # self.cd0 = torch.cat([torch.tensor([-1]).to(self.device), cd0], dim=0) + 1 ## for retreival probability, considering padding
        # self.cd1 = torch.cat([torch.tensor([-1]).to(self.device), cd1], dim=0) + 1 ## for retreival probability, considering padding

        self.i2cd01 = self.i2cd0 * self.num_cluster + self.i2cd1
        self.indices, self.indptr = construct_index(self.i2cd01, self.num_cluster)

        # Number of items under codebook 
        # self.wkk = cd0m.T @ cd1m
        # self.ck2num = (self.indptr[1:] - self.indptr[:-1]).reshape((self.num_cluster, self.num_cluster))
        # self.cd2weights = self.indptr[1:] - self.indptr[:-1]
        
        if bias is not None:
            self.bias = bias.view(-1)
            bias = torch.exp(bias)
            # construct cb2cates
            i2bias = bias.view(-1)  # I
            # Expand dimensions of i2cb to match i2cates
            # i2cd01 = self.i2cd01.repeat(1, T)

            # Initialize cd2weights tensor
            self.cd2weights = torch.zeros(self.num_code, device=i2bias.device, dtype=i2bias.dtype)  # (C0 * C1)

            # Use scatter_add_ to add the values from i2cates to the correct positions in cb2weights
            self.cd2weights.scatter_add_(0, self.i2cd01, i2bias)
            
            # Normalization
            self.cumBias = bias[self.indices].view(-1)  # I
            for c in range(self.num_code):
                start, end = self.indptr[c], self.indptr[c + 1]
                if end > start:
                    cumsum = self.cumBias[start:end].cumsum(dim=0)
                    self.cumBias[start:end] = cumsum / cumsum[-1]
        else:
            self.bias = None
            self.cd2weights = self.indptr[1:] - self.indptr[:-1]  # (C0 * C1)
            
    def get_communication_cost(self):
        return sum(get_size(m) for m in (self.cb0, self.cb1, self.cd2weights, self.indices, self.indptr)) / (1 << 20)