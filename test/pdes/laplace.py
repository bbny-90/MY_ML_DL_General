import inspect

def test_divergence_vector_2d():
    from pdes.operators import get_divergence_vector
    import torch
    import numpy as np
    ndata, ndim = 10, 2
    x = torch.rand(ndata, ndim).float()
    x.requires_grad_(True)
    y = torch.zeros(ndata, ndim)
    y[:, 0] += x[:, 0]**2
    y[:, 1] += x[:, 1]**2
    div_y = get_divergence_vector(y, x).detach().numpy().flatten()
    x = x.detach().numpy()
    div_y_exact = np.sum(x, axis=1) * 2.
    assert np.allclose(div_y, div_y_exact)
    print(f"{inspect.stack()[0][3]} is passed")

def test_laplace_1d():
    from pdes.operators import get_laplace_scalar
    import torch
    import numpy as np
    ndata = 10
    x = torch.rand(ndata, 1).float()
    x.requires_grad_(True)
    y = x**3
    lap_true = (6. * x).detach().numpy()
    lap_autodiff = get_laplace_scalar(y, x).detach().numpy()
    assert np.allclose(lap_autodiff, lap_true)
    print(f"{inspect.stack()[0][3]} is passed")

def test_laplace_3d():
    from pdes.operators import get_laplace_scalar
    import torch
    import numpy as np
    ndata, ndim = 10, 3
    x = torch.rand(ndata, ndim).float()
    x.requires_grad_(True)
    y = x.pow(3).sum(dim=1).view(-1, 1)
    lap_true = 6. * x.sum(dim=1).detach().numpy().reshape(-1, 1)
    lap_autodiff = get_laplace_scalar(y, x).detach().numpy()
    assert np.allclose(lap_autodiff, lap_true)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_divergence_vector_2d()
    test_laplace_1d()
    test_laplace_3d()