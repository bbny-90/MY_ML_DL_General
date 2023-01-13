import os
import pathlib
import pandas as pd
import numpy as np
import torch

pjoin = os.path.join
DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())

class Affine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.teta = torch.nn.Parameter(torch.tensor(0.4)).float().requires_grad_(True)
        self.trnas = torch.nn.Parameter(torch.tensor([[0., 0.]])).float().requires_grad_(True)
    
    def forward(self, x):
        c, s = torch.cos, torch.sin
        rot = torch.zeros(2, 2)
        rot[0, 0] += c(self.teta)
        rot[0, 1] -= s(self.teta)
        rot[1, 0] += s(self.teta)
        rot[1, 1] += c(self.teta)
        xbar = x @ rot.T + self.trnas
        return xbar

def get_data(address):
    df = pd.read_csv(address)[['x', 'y']]
    x = df.to_numpy()
    teta = np.pi / 3.5
    c, s = np.cos, np.sin
    rot = np.array([[c(teta), -s(teta)], [s(teta), c(teta)]])
    trans = np.array([[1.33, -0.4]])
    xbar = x @ rot.T + trans
    return x, xbar, rot, trans