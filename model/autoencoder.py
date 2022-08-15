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
        params: dict, 
    ) -> None:
        self.__input_size:int = params["input_size"]
        self.__hidden_size:int = params["hidden_size"]
        self.__seq_length:int = params["seq_length"]
        self.__num_hid_layer:int = params["num_hid_layer"]
        self.__bias:bool = params["bias"]
        self.__act_type:str = params["act_type"]
        self.__num_rec_layers:int = params["num_rec_layers"]

        self.encoder:ManyToOne = None
        self.decoder:OneToMany = None
        self.__build_models()
    
    
    def __build_models(self) -> None:
        params = {
            'input_size':self.__input_size,
            'hidden_size':self.__hidden_size,
            'seq_length':self.__seq_length,
            'num_hid_layer':self.__num_hid_layer,
            'bias':self.__bias,
            'act_type':self.__act_type,
            'num_rec_layers':self.__num_rec_layers,
        }
        self.encoder = ManyToOne(params)
        params['input_size'] = self.__hidden_size
        params['hidden_size'] = self.__input_size
        params['seq_length'] = self.__seq_length
        self.decoder = OneToMany(params)

    
    def encode(self, x):
        return self.encoder.forward(x)
    
    def decode(self, x):
        return self.decoder.forward(x)
    
    def reconstruct(self, x):
        z = self.encode(x)
        return self.decode(x)