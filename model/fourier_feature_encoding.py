from typing import List
import numpy as np
import torch

class MyActivation(torch.nn.Module):
    def __init__(self, actFun:str):
        super(MyActivation, self).__init__()
        if actFun == "relu":
            self.actFun = torch.nn.ReLU()
        elif actFun == 'elu':
            self.actFun = torch.nn.ELU()
        elif actFun == 'iden':
            self.actFun = lambda x: x
        else:
            raise NotImplementedError(actFun)
    
    def forward(self, x):
        return self.actFun(x)


class GaussianFourierFeatureMapping(torch.nn.Module):
    def __init__(self, input_dim, output_dim, sigma):
        super(GaussianFourierFeatureMapping, self).__init__()
        self.B = torch.randn(size=(input_dim, output_dim)) * sigma * np.pi * 2.
        self.B.requires_grad_(False)
    
    def forward(self, x):
        # x: (b, d)
        assert x.ndim == 2
        x_proj = x @ self.B
        return torch.concat([torch.cos(x_proj), torch.sin(x_proj)], axis=-1)


class MyMLP(torch.nn.Module):
    def __init__(
        self,
        layers:List[int],
        activations:List[MyActivation],
        last_bias=False,
        positional_encoding=False,
        params = {}
        ) -> None:
        """
            layers: [1, 5, 5, 1]
        """
        super(MyMLP, self).__init__()
        assert len(layers) >= 3, len(layers)
        assert len(activations) == len(layers)-1
        net = []
        layers_ = layers
        if positional_encoding:
            net.append(
                GaussianFourierFeatureMapping(layers[0], layers[1], sigma=params['sigma_pos_enc'])
            )
            layers_[0] = layers_[1]*2
        lb = True
        for i in range(len(activations)):
            if i == len(activations):
                lb = last_bias
            net.append(
                torch.nn.Linear(layers_[i], layers_[i+1], bias=lb),
            )
            net.append(activations[i])
        self.net = torch.nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)
    
    def train_regression(self, x, y, opt:torch.optim.Optimizer):
        """
            train one batch data
        """
        loss_func = torch.nn.MSELoss()
        y_ = self.forward(x)
        assert y_.shape == y.shape, y_.shape == y.shape
        loss = loss_func(y_, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()