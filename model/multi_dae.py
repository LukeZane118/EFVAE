import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDAE(nn.Module):
   def __init__(self, args, dataset):
      super().__init__()
      self.q_dims = [dataset.n_items] + args.hidden_dim
      self.p_dims = self.q_dims[::-1]

      # Layers
      self.dims = self.q_dims + self.p_dims[1:]
      self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                   d_in, d_out in zip(self.dims[:-1], self.dims[1:])])

      self.drop = nn.Dropout(args.dropout_prob)

      # Initilization
      self.init_weights(self.layers)

   def forward(self, x):
      h = F.normalize(x, p=2, dim=1)
      h = self.drop(h)
      for i, layer in enumerate(self.layers):
         h = layer(h)
         if i != len(self.layers) - 1:
            h = torch.tanh(h)
      return h

   def init_weights(self, m):
      for layer in m:
         if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias, 0, 0.001)
            
   def loss_function(self, recon_x, x):
      neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, dim=1) * x, -1))
      return neg_ll
