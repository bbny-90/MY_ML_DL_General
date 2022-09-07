import os
import sys
import pathlib
import inspect
import torch
import numpy as np

NET2NET_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
TRANS_LEARN_PATH = os.path.join(NET2NET_PATH, "..")
TEST_PATH = os.path.join(TRANS_LEARN_PATH, "..")
PACK_PATH = os.path.join(TEST_PATH, "..")
sys.path.append(PACK_PATH)

from model.nueral_net_pt import MLP

def test1():
    torch.manual_seed(0)
    params_teacher = {
        "in_dim": 2,
        "out_dim": 2,
        "hid_dim": 3,
        "num_hid_layer": 3,
        "act_type": "relu",
    }

    mlp_teacher = MLP(params=params_teacher)
    for i in mlp_teacher.weight_layer_indices:
        print(mlp_teacher.mlp[i].weight.shape)
    print(mlp_teacher.get_network_architecture())

    mlp_student = mlp_teacher.widen(params_teacher["hid_dim"]*2)

    x = torch.rand(5, 2)
    with torch.no_grad():
        y = mlp_teacher(x).numpy()
        yy = mlp_student(x).numpy()
    assert np.allclose(y, yy)
    assert np.allclose(np.abs(y).sum(), np.abs(yy).sum()), (np.abs(y).sum(), np.abs(yy).sum())
    assert np.allclose(np.abs(y).sum(), 1.8474461), np.abs(y).sum()
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()
