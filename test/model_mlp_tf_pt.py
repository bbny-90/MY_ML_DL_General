import os
import sys
import pathlib
import inspect


DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

def test1():
    """
        to verify tf and pt produce the exact same output
        given the same input setup.
    """
    import tensorflow as tf
    import torch
    from model.nueral_net_tf import MLP as MLPTF
    from model.nueral_net_pt import MLP as MLPPT
    import numpy as np
    bsz, dim = 5, 2
    params = {'in_dim':dim,
              'out_dim':3,
              'hid_dim':5,
              'num_hid_layer':3,
              'act_type':'relu',
             }
    x = np.random.randn(5, 2)
    modelTF = MLPTF(params)
    modelTF.set_weights(np_seed=0)
    modelPT = MLPPT(params)
    modelPT.set_weights(np_seed=0)
    ytf = modelTF(x)
    with torch.no_grad():
        ypt = modelPT(torch.from_numpy(x).float())
    assert np.allclose(tf.norm(ytf), tf.norm(ypt))
    print(f"{inspect.stack()[0][3]} is passed")


if __name__ == "__main__":
    test1()