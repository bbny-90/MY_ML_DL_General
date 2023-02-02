import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PACK_ADD = os.path.join(DIR_PATH, "..")
sys.path.append(PACK_ADD)

def test1():
    import numpy as np
    import torch
    from model.fourier_feature_encoding import (
        MyMLP,
        MyActivation
    )
    import matplotlib.pyplot as plt
    torch.manual_seed(66)
    x = torch.rand(100, 1).float().detach()
    y = torch.sin(x * np.pi * 7).detach()

    
    model = MyMLP(
        layers=[1, 40, 20, 1],
        activations=[MyActivation(i) for i in ['relu', 'relu', 'iden']],
        positional_encoding=True,
        params={'sigma_pos_enc':2}
        )
    opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    for e in range(700):
        loss = model.train_regression(x, y, opt)
        print(e, loss)
    assert np.allclose(loss, 6.2618969423056114e-06)
    if 0:
        plt.scatter(x, y)
        plt.scatter(x, model(x).detach())
        plt.show()
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()