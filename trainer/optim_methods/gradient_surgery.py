import copy
import random
from typing import List, Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class PCGrad():
    def __init__(self, 
            optimizer:Optimizer, 
            need_surgery=True,
            lr_scheduler:Optional[ReduceLROnPlateau]=None,
        ):
        """
            this is a multi-objective optimizer that aims to suppress the gradient conflicts
            based on the following paper
            Gradient Surgery for Multi-Task Learning
            T. Yu, et al 2020

        """
        self._optim = optimizer
        self._lr_scheduler = lr_scheduler
        self.need_surgery = need_surgery

    @property
    def optimizer(self):
        return self._optim

    def step_lr_scheduler(self, val):
        self._lr_scheduler.step(val)

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def backward_regular(self, 
        objectives:List[torch.Tensor], 
        objectives_weights: List[float]
        )-> None:
        assert len(objectives) == len(objectives_weights),\
            (len(objectives) == len(objectives_weights))
        loss = torch.tensor(0.).to(objectives[0].device)
        for obj, w in zip(objectives, objectives_weights):
            loss += (obj.mul(w))
        loss.backward()


    def backward_surgery(self, objectives:List[torch.Tensor]):
        grads, shapes, trainables = self._get_grad_params_wrt_all_objective(objectives)
        pc_grad = self._project_conflict(grads, trainables)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)


    def _project_conflict(self, 
            grads:List[torch.Tensor],
            trainables: List[torch.Tensor]
        )->torch.Tensor:
        """
            here we implement alog 1 in page 4 of the following paper
            https://arxiv.org/pdf/2001.06782.pdf
        """
        shared = torch.stack(trainables).prod(0).bool()
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2) # TODO: avoid zero division
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack(
            [g[shared] for g in pc_grad]).mean(dim=0
        )
        merged_grad[~shared] = torch.stack(
            [g[~shared] for g in pc_grad]).sum(dim=0
        )
        return merged_grad

    def _set_grad(self, grads):
        '''
            set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _get_grad_params_wrt_all_objective(self, objectives: List[torch.Tensor]
        )->Tuple[List[torch.Tensor], List[Tuple[int, int]], List[torch.Tensor]]:
        grads, shapes, trainables = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            grad, shape, trainable = self._get_grad_params_wrt_objective(obj)
            flatten_grad = torch.cat([g.flatten() for g in grad])
            flatten_trainable = torch.cat([g.flatten() for g in trainable])
            grads.append(flatten_grad)
            trainables.append(flatten_trainable)
            shapes.append(shape)
        return grads, shapes, trainables

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def _get_grad_params_wrt_objective(self, obj:torch.Tensor
        )-> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[bool]]:
        '''
            get the gradient of the parameters of the network wrt a specific objective
            
            output:
            - grad: gradient of the parameters
            - shape: shape of the parameters
            - trainable: whether the parameter is trainable
        '''
        assert obj.dim() == 0, obj.dim()
        obj.backward(retain_graph=True)
        grad, shape, trainable = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: # in case of multi-head
                    grad.append(torch.zeros_like(p).to(p.device))
                    shape.append(p.shape)
                    trainable.append(torch.zeros_like(p).to(p.device))
                    continue
                grad.append(p.grad.clone())
                shape.append(p.grad.shape)
                trainable.append(torch.ones_like(p).to(p.device))
        return grad, shape, trainable