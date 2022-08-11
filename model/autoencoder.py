from abc import ABC, abstractmethod
import torch
from .rnn_pt import ManyToOne, OneToMany

class AEBase(ABC):
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
    
    @abstractmethod
    def encode(self, x: torch.tensor):
        pass
    
    @abstractmethod
    def decode(self, x: torch.tensor):
        pass

class AERNN(AEBase):
    def __init__(self,
                 params_encoder: dict, 
                 params_decoder: dict) -> None:
        self.encoder = ManyToOne(params_encoder)
        self.decoder = OneToMany(params_decoder)
    
    def encode(self, x):
        return self.encoder.forward(x)
    
    def decode(self, x):
        return self.decoder.forward(x)