from typing import List
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from model.nueral_net_pt import MLP

class VariationalAE(nn.Module):
    def __init__(self, 
        device,
        x_dim=2,
        latent_dim=2,
        hidden_dim=10,
        num_hid_layer = 3,
        ):
        super(VariationalAE, self).__init__()
        params = {
            "in_dim":x_dim,
            "out_dim":latent_dim*2,# mean and var
            "hid_dim":hidden_dim,
            "num_hid_layer":num_hid_layer,
            "act_type":'relu'
        }
        self._enc_fun = MLP(params=params)
        params = {
            "in_dim":latent_dim,
            "out_dim":x_dim,
            "hid_dim":hidden_dim,
            "num_hid_layer":num_hid_layer,
            "act_type":'relu'
        }
        self._dec_fun = MLP(params=params)
        
        self.latent_dim = latent_dim
        self.sample_dist = torch.distributions.Normal(0, 1)
        if device.type != 'cpu':
            # a trick to sample on GPU
            self.sample_dist.loc = self.sample_dist.loc.cuda()
            self.sample_dist.scale = self.sample_dist.scale.cuda()
        self.kl = 0.

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        tmp = self._enc_fun(x)
        mu = tmp[:, :self.latent_dim]
        logvar = tmp[:, self.latent_dim:]
        return mu, logvar
    
    def decode(self, z):
        return self._dec_fun(z)
    
    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        input_ = self.decode(z)
        return mu, log_var, z, input_

    def loss_function(self,input, mu, log_var, recons, kld_weight) -> dict:
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss, {'recon':recons_loss.detach(), 'KLD':-kld_loss.detach()}
