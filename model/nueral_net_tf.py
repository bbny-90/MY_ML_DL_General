import io
import os
import json
import logging
from tensorflow import nn
from tensorflow import __version__ as tf_version
from tensorflow.keras.layers import Dense
from tensorflow.keras import (Sequential, 
                              Model)


class MLP(Model):
    def __init__(self, 
            params: dict, 
        ) -> None:
        super().__init__()
        self.in_dim = int(params['in_dim'])
        self.out_dim = int(params['out_dim'])
        self.hid_dim = int(params['hid_dim'])
        self.num_hid_layer = int(params['num_hid_layer'])
        self.act_type: str = params['act_type']
        if self.act_type == "relu":
            actFun = nn.relu
        else:
            raise NotImplementedError(f"activation {self.act_type} is not supported")
        
        model = [Dense(self.hid_dim, activation=actFun, input_shape=(self.in_dim, ))]
        for _ in range(self.num_hid_layer):
            model.append(Dense(self.hid_dim, activation=actFun))
        model.append(Dense(self.out_dim))
        
        model = Sequential(model)
        self.net = model
        logging.info("model initilized!")

    def load_weights(self, nn_weights_path: str):
        assert os.path.exists(nn_weights_path), f"{nn_weights_path} doesnt exist"
        self.net.load_weights(nn_weights_path)
        logging.info("model loaded!")

    def call(self, x):
        return self.net(x)

    def save(self, dir_to_save: str, 
             model_info_name: str, 
             weight_name: str) -> None:
        weight_name_ = weight_name
        if not weight_name_.endswith(".h5"):
            weight_name_ = weight_name_ + ".h5"
        weight_path = os.path.join(dir_to_save, weight_name_)
        self.net.save_weights(weight_path)

        tmp = {"weight_path": weight_path}
        for k, v in self.__dict__.items():
            if k in {"in_dim", "out_dim", "hid_dim", "num_hid_layer", "act_type"}:
                tmp[k] = v
        tmp["tf_version"] = tf_version
        tmp["model"] = "MLP"
        
        tmp_name_ = model_info_name
        if not tmp_name_.endswith(".json"):
            tmp_name_ = tmp_name_ + ".json"
        with open(os.path.join(dir_to_save, tmp_name_), "w") as f:
            json.dump(tmp, f)
        print("model saved!")