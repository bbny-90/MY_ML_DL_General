import io
import h5py
import os
import json
import logging
import numpy as np
from tensorflow import nn
from tensorflow import __version__ as tf_version
from tensorflow.keras.layers import Dense
from tensorflow.keras import (Sequential, 
                              Model)
from helper.s3_client import S3BucketClient


class MLP(Model):
    def __init__(self, 
            params: dict, 
        ) -> None:
        """
            inputs:
                params = {"in_dim":
                          "out_dim":
                          "hid_dim":
                          "num_hid_layer":
                          "act_type":
                }
            ------
            e.g., a network with 3 hidden layers: in->h1->h2->h3->out
        """
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
        for _ in range(self.num_hid_layer - 1):
            model.append(Dense(self.hid_dim, activation=actFun))
        model.append(Dense(self.out_dim))
        
        model = Sequential(model)
        self.net = model
        logging.info("model initilized!")

    def set_weights(self, np_seed:int=None, method=str()):
        """
            the method is not consistent with the litrature
            it is used for checking with tf implemetation
        """
        if np_seed is not None:
            np.random.seed(np_seed)
        for i, layer in enumerate(self.net.layers):
            w, b = layer.get_weights()
            ww = np.random.randn(w.shape[1], w.shape[0]) / np.sqrt(max(w.shape))
            bb = np.random.randn(b.shape[0]) / np.sqrt(b.shape[0])
            self.net.layers[i].set_weights([ww.T,bb])

    def load_weights(self, nn_weights_path: str):
        assert os.path.exists(nn_weights_path), f"{nn_weights_path} doesnt exist"
        self.net.load_weights(nn_weights_path)
        logging.info("model loaded!")

    def load_weights_to_s3(self, s3client: S3BucketClient, path: str):
        wbytes = s3client.get_object(path=path)
        with io.BytesIO(wbytes) as f:
            with h5py.File(f, "r") as h:
                self.net.load_weights(wbytes)
                # self.net = tf.keras.models.load_model(h, custom_objects={"MLP": MLP})
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
    
    def save_to_s3(
        self,
        s3_bucket_client: S3BucketClient,
        path: str
    ) -> None:
        with io.BytesIO() as model_io:
            # Instanciate h5 file using the io.
            with h5py.File(model_io, "w") as model_h5:
                # Save the Keras model to h5 object (and indirectly to bytesio).
                self.net.save(model_h5)
                # Make sure the data is written entirely to the bytesio object.
                model_h5.flush()
            # Upload to S3.
            s3_bucket_client.save_object(path, model_io.getvalue())

        # for metadata
        # byte_value = bytes(json.dumps(self.net.to_json()).encode("UTF-8"))
        # # byte_value = bytes(json.dumps(self.populate_model_metadata()).encode("UTF-8"))
        # s3_bucket_client.save_object(path, byte_value)
        # del byte_value
        logging.info("model saved")