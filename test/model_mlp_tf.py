import os
import sys
import shutil
import pathlib
import inspect


DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

def test1():
    import tensorflow as tf
    from model.nueral_net_tf import MLP
    import numpy as np
    bsz, dim = 5, 2
    params = {'in_dim':dim,
              'out_dim':3,
              'hid_dim':5,
              'num_hid_layer':3,
              'act_type':'relu',
             }
    model = MLP(params)
    # x = tf.random.uniform((5, 2))
    x = np.random.randn(5, 2)
    y = model(x)
    wdir = os.path.join(DIR_PATH, ".tmp")
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    wname = "xxx"
    model.save(wdir, wname, wname)
    del model

    model = MLP(params)
    wadd = os.path.join(wdir, wname+".h5")
    model.load_weights(wadd)
    y_ = model(x)
    assert np.allclose(tf.norm(y), tf.norm(y_))
    assert x.shape == (5, 2) 
    assert y.shape == (5, 3)
    shutil.rmtree(wdir)
    print(f"{inspect.stack()[0][3]} is passed")

def test2():
    import numpy as np
    import tensorflow as tf
    from model.nueral_net_tf import MLP
    from trainer.nueral_net_tf import train_encoder_decoder
    tf.random.set_seed(0)
    np.random.seed(0)

    dim = 3
    params_enc = {'in_dim':dim,
              'out_dim':3,
              'hid_dim':5,
              'num_hid_layer':3,
              'act_type':'relu',
    }
    params_dec = {'in_dim':3,
              'out_dim':dim,
              'hid_dim':5,
              'num_hid_layer':3,
              'act_type':'relu',
    }
    train_params = {
        "batch_size": 2,
        "drop_last_dl": False,
        "lr": 0.001,
        "epochs": 30,
        "loss_fun": "mse",
        "optimizer": "ADAM",
    }
    
    enc = MLP(params_enc)
    dec = MLP(params_dec)
    xtrn = np.random.randn(10, dim)
    loss = train_encoder_decoder(enc, dec, xtrn, train_params)
    np.allclose(loss['recn_mse_train'][0], 1.003762800)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()
    test2()