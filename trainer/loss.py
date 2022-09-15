import numpy as np
import torch

def CrossEntropyLoss(h_pred:torch.Tensor, label:torch.Tensor):
    loss = -h_pred[np.arange(label.shape[0]), label]+\
            torch.log(torch.exp(h_pred).sum(dim=1))
    return loss.mean()