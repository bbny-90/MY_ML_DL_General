import os
import sys
import shutil
import pathlib
import inspect


DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PACK_ADD = os.path.join(DIR_PATH, "..")
sys.path.append(PACK_ADD)

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
    from model.autoencoder_tf import AEMLP
    from trainer.nueral_net_tf import train_encoder_decoder
    tf.random.set_seed(0)
    np.random.seed(0)

    dim = 3
    params_ae = {'input_size':dim,
                  'hidden_size':30,
                  'num_hid_layer':3,
                  'latent_size':30,
                  'act_type':'relu',
                 }
    train_params = {
        "batchsize": 2,
        "drop_last_dl": False,
        "lr": 0.001,
        "epochs": 30,
        "loss_fun": "mse",
        "optimizer": "ADAM",
    }
    
    ae_model = AEMLP(params_ae)
    xtrn = np.random.randn(10, dim)
    loss = train_encoder_decoder(ae_model, xtrn, train_params)
    np.allclose(loss['recn_mse_train'][0], 1.003762800)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()
    test2()