from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch
from .rnn_pt import ManyToOne, OneToMany
from .nueral_net_pt import MLP

class AEBase(ABC):
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
    
    @abstractmethod
    def encode(self, 
        x: torch.tensor
    ) -> torch.tensor:
        pass
    
    @abstractmethod
    def decode(self, 
        x: torch.tensor
    ) -> torch.tensor:
        pass
    
    def reconstruct(self, 
        x: Union[torch.tensor, np.ndarray]
    ) -> Union[torch.tensor, np.ndarray]:
        isnp = False
        xpt = x
        if isinstance(x, np.ndarray):
            isnp = True
            xpt = torch.from_numpy(x).float()
        x_ = self.decode(self.encode(xpt))
        if isnp:
            x_ = x_.detach().numpy()
        return x_


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

    
    def encode(self, 
        x: torch.tensor
    ) -> torch.tensor:
        return self.encoder.forward(x)
    
    def decode(self,
        x: torch.tensor
    ) -> torch.tensor:
        return self.decoder.forward(x)

class AEMLP(AEBase):
    def __init__(self,
        params: dict, 
    ) -> None:
        self.__input_size:int = params["input_size"]
        self.__hidden_size:int = params["hidden_size"]
        self.__num_hid_layer:int = params["num_hid_layer"]
        self.__latent_size:int = params["latent_size"]
        self.__act_type:str = params["act_type"]

        self.encoder:MLP = None
        self.decoder:MLP = None
        self.__build_models()
    
    
    def __build_models(self) -> None:
        params = {
            'in_dim':self.__input_size,
            'hid_dim':self.__hidden_size,
            'num_hid_layer':self.__num_hid_layer,
            'out_dim':self.__latent_size,
            'act_type':self.__act_type,
        }
        self.encoder = MLP(params)
        params['in_dim'] = self.__latent_size
        params['out_dim'] = self.__input_size
        self.decoder = MLP(params)

    
    def encode(self, 
        x: torch.tensor
    ) -> torch.tensor:
        return self.encoder.forward(x)
    
    def decode(self, 
        x: torch.tensor
    ) -> torch.tensor:
        return self.decoder.forward(x)