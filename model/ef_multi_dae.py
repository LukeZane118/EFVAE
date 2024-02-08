import torch
import torch.nn as nn
import torch.nn.functional as F


class EFMultiDAE(nn.Module):
   def __init__(self, args, dataset):
      super().__init__()
      self.q_dims = [dataset.n_items] + args.hidden_dim
      self.p_dims = self.q_dims[::-1]

      # Layers
      self.dims = self.q_dims + self.p_dims[1:]
      self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                   d_in, d_out in zip(self.dims[1:-2], self.dims[2:-1])])

      # Encoder Embedding & Bias
      self.enc_embeddings = nn.Embedding(dataset.n_items, self.dims[1])
      self.enc_bias = nn.Parameter(torch.zeros((1, self.dims[1])))
      
      # Decoder Embedding
      self.dec_embeddings = nn.Embedding(dataset.n_items, self.dims[1])
      self.dec_bias = nn.Embedding(dataset.n_items, 1)

      self.drop1d = nn.Dropout(args.dropout_prob)
      self.drop2d = nn.Dropout2d(args.dropout_prob)
      
      # Initilization
      for m in (self.enc_embeddings, self.dec_embeddings, self.layers):
         self.init_weights(m)
      torch.nn.init.normal_(self.enc_bias.data, 0, 0.001)
      torch.nn.init.normal_(self.dec_bias.weight.data, 0, 0.001)

   def forward(self, x):
      x = F.normalize(x, p=2, dim=1)
      x = self.drop1d(x)
      h = x @ self.enc_embeddings.weight + self.enc_bias
      h = torch.tanh(h)
      for i, layer in enumerate(self.layers):
         h = layer(h)
         h = torch.tanh(h)
      recon_x = h @ self.dec_embeddings.weight.T + self.dec_bias.weight.T
      return recon_x

   def init_weights(self, m):
        if isinstance(m, nn.Embedding):
            torch.nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias.data, 0, 0.001)
            
   def loss_function(self, neg_logit, neg_log_prob, pos_logit, pos_log_prob):
      # idx_mtx = (pos_logit != 0).double()
      new_pos = pos_logit - pos_log_prob.detach()
      new_neg = neg_logit - neg_log_prob.detach()

      neg_log_sum_exp = torch.logsumexp(new_neg, dim=-1, keepdim=True)
      final = torch.logaddexp(new_pos, neg_log_sum_exp)
      
      # return torch.sum((- new_pos + final) * idx_mtx, dim=-1).mean()
      return torch.sum(-new_pos + final, dim=-1).mean()

   def calculate_loss(self, pos_items, sampler):
        
      h = self.enc_embeddings(pos_items) # batch_user * cur_max_len * dims
      h = self.drop2d(h)
      h = torch.sum(h, dim=1) / (pos_items.shape[-1] ** 0.5 + 1e-8) + self.enc_bias
      h = torch.tanh(h)
      
      for i, layer in enumerate(self.layers):
         h = layer(h)
         h = torch.tanh(h)
      
      with torch.no_grad():
         pos_log_prob, neg_items, neg_log_prob = sampler(h, pos_items)
         
      pos_items_emb = self.dec_embeddings(pos_items)
      neg_items_emb = self.dec_embeddings(neg_items)
      
      pos_items_bias = self.dec_bias(pos_items).squeeze(-1)
      neg_items_bias = self.dec_bias(neg_items).squeeze(-1)
      
      pos_logit = h @ pos_items_emb.permute(0, 2, 1) + pos_items_bias
      neg_logit = h @ neg_items_emb.permute(0, 2, 1) + neg_items_bias
      
      ce_loss = self.loss_function(neg_logit, neg_log_prob, pos_logit, pos_log_prob)

      loss = ce_loss
      
      return loss, neg_items.squeeze()