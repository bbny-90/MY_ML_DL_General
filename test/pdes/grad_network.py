import inspect


def test_first_grad_scalar():
    from pdes.operators import get_grad_scalar
    import torch
    import numpy as np
    ndata = 10
    x = torch.rand(ndata, 1).float()
    x.requires_grad_(True)
    
    class MLP():
        def __init__(self) -> None:
            self.layer1 = torch.nn.Linear(1, 10, bias=False)
            self.layer2 = torch.nn.Linear(10, 1, bias=False)
        
        def __call__(self, x) -> torch.tensor:
            y = self.layer1(x)
            y = self.layer2(y)
            return y

    model = MLP()
    y = model(x)
    dydx_autodiff = get_grad_scalar(y, x).detach().numpy()
    with torch.no_grad():
        dydx_true = torch.matmul(model.layer2.weight.data, model.layer1.weight.data).numpy()
    
    assert np.allclose(dydx_autodiff, dydx_true)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_first_grad_scalar()