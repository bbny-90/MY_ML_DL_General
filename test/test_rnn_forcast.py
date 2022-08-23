from operator import mod
import os
import sys
import pathlib
import inspect

import torch

DIR_PATH = os.path.abspath(pathlib.Path(__file__).parent.absolute())
PROB_FILE_GEN = os.path.join(DIR_PATH, "..")
sys.path.append(PROB_FILE_GEN)

import numpy as np
import matplotlib.pyplot as plt
from helper.data_processor import (create_window_forcast_data,
                                    shuffle)

from model.rnn_pt import ManyToOne

w_size, num_future = 5, 1
num_tst = 40
params_rnn = {
    'input_size':1,
    'hidden_size':1,
    'seq_length':w_size,
    'bias':False,
    'act_type':'relu',
    'num_rec_layers':2,
}


def get_data():
    x = np.linspace(0., 799., 800)
    x = np.sin(x * 2. * 3.1416/40)
    return x
x = get_data().reshape(-1, 1)

# plt.plot(x[:,0], x[:, 1])
# plt.show()

xwn, ywn = create_window_forcast_data(x, w_size, num_future)
xtrn, ytrn = xwn[:-num_tst, ...], ywn[:-num_tst, ...]
xtst, ytst = xwn[-num_tst:, ...], ywn[-num_tst:, ...]

to_torch = lambda x: torch.from_numpy(x).float()
xtrn = to_torch(xtrn)
print(xtrn.shape)

model = ManyToOne(params_rnn)
print(model(xtrn).shape)


