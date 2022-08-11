from operator import mod
import os
import sys
import pathlib
import inspect

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

def test_rnn():
    import torch
    from model.autoencoder import AERNN
    bsz= 10
    params_enc = {'input_size':5,
                  'hidden_size':2,
                  'seq_length':5,
                  'num_hid_layer':3,
                  'bias': False,
                  'act_type':'relu',
                  'num_rec_layers':3,
                 }
    params_dec = {'input_size':params_enc['hidden_size'],
                  'hidden_size':params_enc['input_size'],
                  'seq_length':params_enc['seq_length'],
                  'num_hid_layer':3,
                  'bias': False,
                  'act_type':'relu',
                  'num_rec_layers':3,
                 }
    with torch.no_grad():
        x = torch.rand(bsz, 
                       params_enc['seq_length'], 
                       params_enc['input_size'])
        model = AERNN(params_enc, params_dec)
        enc = model.encode(x)
        x_ = model.decode(enc)
        print(enc.shape)
        print(x_.shape)
    assert x_.shape == x.shape
    assert enc.shape == (x.shape[0], params_enc['hidden_size'])
    print(f"{inspect.stack()[0][3]} is passed")

def test_rnn_train():
    import torch
    import numpy as np
    from model.autoencoder import AERNN
    from trainer.nueral_net_pt import train_encoder_decoder
    bsz= 5
    params_enc = {'input_size':2,
                  'hidden_size':30,
                  'seq_length':3,
                  'num_hid_layer':3,
                  'bias': False,
                  'act_type':'relu',
                  'num_rec_layers':3,
                 }
    params_dec = {'input_size':params_enc['hidden_size'],
                  'hidden_size':params_enc['input_size'],
                  'seq_length':params_enc['seq_length'],
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
    x = np.random.rand(bsz, 
                    params_enc['seq_length'], 
                    params_enc['input_size'])    
    
    dev = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
    model = AERNN(params_enc, params_dec)
    loss = train_encoder_decoder(model, x, train_params, dev)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test_rnn()
    test_rnn_train()