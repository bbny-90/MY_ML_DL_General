from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import tensorflow as tf
from .nueral_net_tf import MLP

class AEBase(ABC):
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
    
    @abstractmethod
    def encode(self, 
        x: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        pass
    
    @abstractmethod
    def decode(self, 
        x: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        pass
    
    def reconstruct(self, 
        x: Union[tf.Tensor, np.ndarray]
    ) -> Union[tf.Tensor, np.ndarray]:
        x_ = self.decode(self.encode(x))
        if isinstance(x, np.ndarray):
            x_ = x_.numpy()
        return x_


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
        x: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        return self.encoder(x)
    
    def decode(self, 
        x: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        return self.decoder(x)
    
