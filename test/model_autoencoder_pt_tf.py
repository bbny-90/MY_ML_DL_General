import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PACK_ADD = os.path.join(DIR_PATH, "..")
sys.path.append(PACK_ADD)

def test_mlp_tf_pt():
    """
        to verify the loss and its gradient are exactly the same
        for tf and pt implementation given the same input setup
    """
    import torch
    import numpy as np
    np.random.seed(0)
    from model.autoencoder_pt import AEMLP as AEMLPPT
    from model.autoencoder_tf import AEMLP as AEMLPTF
    from trainer.nueral_net_pt import train_encoder_decoder as trnEDPT
    from trainer.nueral_net_tf import train_encoder_decoder as trnEDTF
    bsz= 5
    params_ae = {'input_size':2,
                  'hidden_size':30,
                  'num_hid_layer':3,
                  'latent_size':30,
                  'act_type':'relu',
                 }
    train_params = {
        "batchsize":bsz,
        "lr":0.01,
        "epochs":10,
        "optimizer":"SGD",
        "drop_last_dl": False,
        "loss_fun":"mse"
    }
    x = np.random.rand(
        bsz,
        params_ae['input_size']
    )
    
    modelPT = AEMLPPT(params_ae)
    modelPT.encoder.set_weights(np_seed=0)
    modelPT.decoder.set_weights(np_seed=1)

    modelTF = AEMLPTF(params_ae)
    modelTF.encoder.set_weights(np_seed=0)
    modelTF.decoder.set_weights(np_seed=1)

    xRecTF = modelTF.reconstruct(x)
    xRecPT = modelPT.reconstruct(x)
    assert np.allclose(xRecTF, xRecPT)
    dev = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
    lossPT = trnEDPT(modelPT, x, train_params, dev)
    lossTF = trnEDTF(modelTF, x, train_params)
    assert np.allclose(
        lossTF['recn_mse_train'], lossPT['recn_mse_train']
    )
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_mlp_tf_pt()