import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PACK_ADD = os.path.join(DIR_PATH, "..")
sys.path.append(PACK_ADD)

def test_rnn():
    import torch
    from model.autoencoder_pt import AERNN
    bsz= 10
    params = {
        'input_size':5,
        'hidden_size':2,
        'seq_length':5,
        'num_hid_layer':3,
        'bias': False,
        'act_type':'relu',
        'num_rec_layers':3,
    }
    with torch.no_grad():
        x = torch.rand(
            bsz, 
            params['seq_length'], 
            params['input_size']
        )
        model = AERNN(params)
        enc = model.encode(x)
        x_ = model.decode(enc)
        print(enc.shape)
        print(x_.shape)
    assert x_.shape == x.shape
    assert enc.shape == (x.shape[0], params['hidden_size'])
    print(f"{inspect.stack()[0][3]} is passed")

def test_rnn_train():
    import torch
    import numpy as np
    from model.autoencoder_pt import AERNN
    from trainer.nueral_net_pt import train_encoder_decoder
    bsz= 5
    params_ae = {'input_size':2,
                  'hidden_size':30,
                  'seq_length':3,
                  'num_hid_layer':3,
                  'bias': False,
                  'act_type':'relu',
                  'num_rec_layers':3,
                 }
    train_params = {
        "batchsize":bsz,
        "lr":0.005,
        "epochs":5000,
        "optimizer":"ADAM",
    }
    x = np.random.rand(
        bsz, 
        params_ae['seq_length'], 
        params_ae['input_size']
    )
    
    dev = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
    model = AERNN(params_ae)
    loss = train_encoder_decoder(model, x, train_params, dev)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_rnn()
    test_rnn_train()