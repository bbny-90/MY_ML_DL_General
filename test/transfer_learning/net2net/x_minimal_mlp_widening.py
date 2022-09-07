import torch
import numpy as np
num_hid_layers = 2
indim, hdim, outdim = 2, 3, 2
hdim_widen = hdim + 2

class MLP():
    def __init__(self, indim, hdim, outdim, num_hid_layers) -> None:
        # in-h1-h2-h3-out -> 3 hid layers, 4 matrices
        arct = [indim] + [hdim] * num_hid_layers + [outdim]
        self.layers = []
        for i in range(len(arct)-1):
            self.layers.append(torch.nn.Linear(arct[i], arct[i+1], bias=True))
        self.arct = arct
    
    def __call__(self, x):
        out = self.layers[0](x)
        i = 1
        while i <len(self.layers):
            out = self.layers[i](out)
            i += 1
        return out
    
    def get_hidden_states(self, x):
        out =  []
        x_ = x*1
        for layer in self.layers:
            out.append(layer(x_))
            x_ = layer(x_)
        return out


    


model = MLP(indim, hdim, outdim, num_hid_layers)


x = torch.rand(5, indim)
y = model(x)
def get_remap_array(hold, hnew):
    assert hold <= hnew, (hold, hnew)
    remap_arr = np.arange(hnew)
    remap_arr[hold:] = np.random.choice(hold, hnew - hold)
    counts = np.zeros(hold, dtype=int)
    for i in remap_arr:
        counts[i] += 1
    return remap_arr, counts

def get_remap_layers(arct_old, arct_new):
    """
        this includes input and output layers as well!
    """
    assert len(arct_old) == len(arct_new), (arct_old, arct_new)
    remap_all = {}
    i = 0
    for hold, hnew in zip(arct_old, arct_new):
        remap_arr, counts = get_remap_array(hold, hnew)
        remap_all[i] = {"remap_arr":remap_arr, "counts":counts}
        i += 1
    return remap_all



model_widen = MLP(indim, hdim_widen, outdim, num_hid_layers)
remap_all = get_remap_layers(model.arct, model_widen.arct)

for i in range(len(model_widen.layers)):
    W = model_widen.layers[i].weight.data
    g = remap_all[i]['remap_arr']
    gg = remap_all[i+1]['remap_arr']
    counts = remap_all[i]['counts']
    for k in range(W.shape[1]):
        for j in range(W.shape[0]):
            model_widen.layers[i].weight.data[j, k] = model.layers[i].weight.data[gg[j], g[k]] / counts[g[k]]

    g = remap_all[i+1]['remap_arr']
    counts = remap_all[i+1]['counts']
    for k in range(W.shape[0]):
        model_widen.layers[i].bias.data[k] = model.layers[i].bias.data[g[k]]# / counts[g[k]]

ynew = model_widen(x)
print(y)
print(ynew - y)