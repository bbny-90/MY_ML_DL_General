import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PACK_ADD = os.path.join(DIR_PATH, "..")
sys.path.append(PACK_ADD)

def test_cross_entropy():
    import numpy as np
    import torch
    from trainer.loss import CrossEntropyLoss as MyCrossEntropyLoss
    ndata, nclass = 5, 3
    label = torch.randint(0, nclass, [ndata,])
    h = torch.rand(ndata, nclass)
    my_loss = MyCrossEntropyLoss(h, label).numpy()
    torch_loss = torch.nn.CrossEntropyLoss()(h, label).numpy()
    assert np.allclose(my_loss, torch_loss)
    print(f"{inspect.stack()[0][3]} is passed")
if __name__ == "__main__":
    test_cross_entropy()