import os
import sys
import pathlib

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

def test_many_to_many():
    import torch
    from model.rnn import ManyToMany
    bsz= 10
    params = {'input_size':5,
              'hidden_size':2,
              'seq_length':5,
              'num_hid_layer':3,
              'bias': False,
              'act_type':'relu',
              'num_rec_layers':3,
             }
    with torch.no_grad():
        x = torch.rand(bsz, 
                       params['seq_length'], 
                       params['input_size'])
        model = ManyToMany(params)
        y = model(x)
        print(y.shape)
    assert y.shape == (bsz, params['seq_length'], params['hidden_size']), y.shape

def test_many_to_one():
    import torch
    from model.rnn import ManyToOne
    bsz= 10
    params = {'input_size':5,
              'hidden_size':2,
              'seq_length':5,
              'num_hid_layer':3,
              'bias': False,
              'act_type':'relu',
              'num_rec_layers':3,
             }
    with torch.no_grad():
        x = torch.rand(bsz, 
                       params['seq_length'], 
                       params['input_size'])
        model = ManyToOne(params)
        y = model(x)
        print(y.shape)
    assert y.shape == (bsz, params['hidden_size']), y.shape

def test_one_to_many():
    import torch
    from model.rnn import OneToMany
    bsz= 10
    params = {'input_size':5,
              'hidden_size':2,
              'seq_length':5,
              'num_hid_layer':3,
              'bias': False,
              'act_type':'relu',
              'num_rec_layers':3,
             }
    with torch.no_grad():
        x = torch.rand(bsz, 
                       1, 
                       params['input_size'])
        model = OneToMany(params)
        y = model(x)
        print(y.shape)
    assert y.shape == (bsz, params['input_size'], params['hidden_size']), y.shape

if __name__ == "__main__":
    test_many_to_many()
    test_many_to_one()
    test_one_to_many()