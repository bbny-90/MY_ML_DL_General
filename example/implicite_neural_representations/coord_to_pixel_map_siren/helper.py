import numpy as np
import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize
)
from PIL import Image

def getImageSpatialCoord(
    numPix:int, 
    dim=2
    ) -> torch.Tensor:
    '''
        generate spatial coordinates for an image
    '''
    tmp = tuple(dim * [torch.linspace(-1, 1, steps=numPix)])
    grid = torch.stack(torch.meshgrid(*tmp), dim=-1).reshape(-1, dim)
    return grid

def transformImage(
    image:np.ndarray,
    targetSize,
    ) -> torch.Tensor:
    img = Image.fromarray(image)
    transform = Compose([
        Resize(targetSize),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

@torch.no_grad()
def getLinearLayer(inFeat, outFeat, initType:int, omega=30.):
    """
        see the derivation in:
        Implicit Neural Representations with Periodic Activation Functions
        Sitzmann et al, NeurIPS 2020
    """
    linear = nn.Linear(inFeat, outFeat)
    if initType == 0:# for the first layer
        tmp = 1. / inFeat
    else:
        tmp = np.sqrt(6. / inFeat) / omega
    linear.weight.uniform_(-tmp, tmp)
    return linear

class Siren(nn.Module):
    def __init__(self, 
        inFeat:int, 
        outFeat:int,
        hidFeat:int,
        numHidLayers:int, 
        omegas:list, # except last layer [30]
        ):
        super().__init__()
        assert len(omegas) == numHidLayers
        net = []
        self.omegas = omegas
        net.append(
            getLinearLayer(inFeat, hidFeat, initType=0, omega=omegas[0])
        )
        for i in range(numHidLayers-1):
            net.append(
                getLinearLayer(hidFeat, hidFeat, initType=1, omega=omegas[1+i])
            )
        # last layer is a simple linear layer
        net.append(
            nn.Linear(hidFeat, outFeat)
        )
        self.fcLayers = nn.Sequential(*net)
    
    def forward(self, coords):
        output = coords
        for i in range(len(self.fcLayers)-1):
            output = torch.sin(self.fcLayers[i](output * self.omegas[i]))
        output = self.fcLayers[-1](output)
        return output