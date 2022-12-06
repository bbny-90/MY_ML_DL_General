import torch
from torch import (
    nn, 
    distributions
)
import numpy as np
from numpy import ndarray
from model.nueral_net_pt import MLP

class CouplingLayer(nn.Module):
    def __init__(self,
        mask:ndarray,
        input_dim:int, 
        output_dim:int, 
        hid_dim:int,
        num_hidden_layers:int,
        act_type='relu'
        ):
        super().__init__()
        params = {
            "in_dim":input_dim,
            "out_dim":output_dim,
            "hid_dim":hid_dim,
            "num_hid_layer":num_hidden_layers,
            "act_type":act_type
        }
        self.scale_mlp = MLP(params)
        self.translation_mlp = MLP(params)
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def forward(self, x:torch.tensor):
        x_masked = x * self.mask
        s = torch.tanh(self.scale_mlp(x_masked))
        t = self.translation_mlp(x_masked)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacobian = s.sum(dim=1)
        return y, log_det_jacobian

    def inverse(self, y:torch.tensor):
        y_masked = y * self.mask
        s = torch.tanh(self.scale_mlp(y_masked))
        t = self.translation_mlp(y_masked)
        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)
        return x


class RealNVP(nn.Module):
    def __init__(self,
        prior_z:distributions, 
        input_dim:int, 
        output_dim:int, 
        hid_dim:int, 
        mask:np.ndarray, 
        num_coupl_layers = 6,
        num_mlp_hid_layers=3,
        act_type = 'relu',
        ):
        super().__init__()
        assert num_coupl_layers >= 2
        modules = []
        modules.append(
            CouplingLayer(
                mask=mask, input_dim=input_dim, output_dim=output_dim,
                hid_dim=hid_dim, num_hidden_layers=num_mlp_hid_layers, act_type=act_type
            )
        )
        for _ in range(num_coupl_layers - 2):
            mask = 1. - mask
            modules.append(
                CouplingLayer(
                    mask=mask, input_dim=input_dim, output_dim=output_dim,
                    hid_dim=hid_dim, num_hidden_layers=num_mlp_hid_layers, act_type=act_type
                )
            )
        modules.append(
            CouplingLayer(
                mask=1. - mask, input_dim=input_dim, output_dim=output_dim,
                hid_dim=hid_dim, num_hidden_layers=num_mlp_hid_layers, act_type=act_type
            )
        )
        self.modules_list = nn.ModuleList(modules)
        self.prior_z = prior_z
        
    def forward(self, x:torch.tensor):
        ldj_sum = 0 # sum of log determinant of jacobian
        for module in self.modules_list:
            x, ldj= module(x)
            ldj_sum += ldj
        return x, ldj_sum

    def inverse(self, z:torch.tensor):
        for module in reversed(self.modules_list):
            z = module.inverse(z)
        return z

    @torch.no_grad()
    def sample(self, num_sample:int):
        self.eval()
        z = self.prior_z.sample((num_sample,))
        x = self.inverse(z)
        z = z.numpy()
        x = x.numpy()
        return x, z