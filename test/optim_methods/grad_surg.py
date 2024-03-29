import os
import sys
import pathlib
import inspect
import torch
import numpy as np

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
TEST_ADD = os.path.join(DIR_PATH, "..")
PACK_ADD = os.path.join(TEST_ADD, "..")
sys.path.append(PACK_ADD)

from trainer.optim_methods.gradient_surgery import PCGrad

def test1():
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self, input_dim,output_dim) -> None:
            super().__init__()
            npr = np.random.RandomState(0)
            self.params = torch.nn.Parameter(torch.from_numpy(npr.rand(input_dim,output_dim)).float())
            self.params.requires_grad = True
        
        def forward(self, x):
            return torch.matmul(x, self.params)

    input_dim, output_dim = 2, 1
    ndata = 5
    x = torch.rand(ndata, input_dim)
    y = torch.rand(ndata, output_dim)
    
    model = Model(input_dim, output_dim)
    opt = torch.optim.SGD(params=list(model.parameters()), lr=0.01)
    pcgrad = PCGrad(opt)
    print("Surgery ---------")
    for _ in range(5):
        pcgrad.zero_grad()
        y_pred = model(x)
        loss1 = (y_pred-y).pow(2).mean()
        loss2 = (y_pred-y).pow(3).mean()
        pcgrad.backward_surgery([loss1, loss2])
        pcgrad.step()
        print(loss1.item(), loss2.item())
    assert np.allclose(loss1.item() + loss2.item(), 0.38952691853), loss1.item() + loss2.item()
    print("Regular ---------")
    model = Model(input_dim, output_dim)
    opt = torch.optim.SGD(params=list(model.parameters()), lr=0.01)
    pcgrad = PCGrad(opt)
    for _ in range(5):
        pcgrad.zero_grad()
        y_pred = model(x)
        loss1 = (y_pred-y).pow(2).mean()
        loss2 = (y_pred-y).pow(3).mean()
        pcgrad.backward_regular([loss1, loss2], [1., 1.])
        pcgrad.step()
        print(loss1.item(), loss2.item())
    assert np.allclose(loss1.item() + loss2.item(), 0.354016266763), loss1.item() + loss2.item()
    print(f"{inspect.stack()[0][3]} is passed")

def test2():
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self, input_dim, hdim, output_dim) -> None:
            super().__init__()
            self.lin1 = torch.nn.Linear(input_dim, hdim)
            self.lin2 = torch.nn.Linear(hdim, output_dim)
        
        def forward(self, x):
            y = torch.nn.ReLU()(self.lin1(x))
            y = self.lin2(y)
            return y
        
        def get_params(self):
            out = [self.lin1.weight.view(-1, 1), self.lin1.bias.view(-1, 1)]
            out += [self.lin2.weight.view(-1, 1), self.lin2.bias.view(-1, 1)]
            out = torch.concat(out, axis=0)
            return out

    input_dim, output_dim = 2, 2
    hdim = 3
    ndata = 5
    x = torch.rand(ndata, input_dim)
    y = torch.rand(ndata, output_dim)
    
    model = Model(input_dim, hdim, output_dim)
    opt = torch.optim.SGD(params=list(model.parameters()), lr=0.01)
    pcgrad = PCGrad(opt)
    for _ in range(5):
        pcgrad.zero_grad()
        y_pred = model(x)
        loss1 = (y_pred-y).pow(2).sum(1).mean()
        loss2 = model.get_params().pow(2).mean()
        pcgrad.backward_surgery([loss1, loss2])
        pcgrad.step()
        print(loss1.item(), loss2.item())
    assert np.allclose(loss1.item() + loss2.item(), 2.2603778690099), loss1.item() + loss2.item()
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()
    test2()
