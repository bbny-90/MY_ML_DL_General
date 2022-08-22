import os
import json
import torch
import numpy as np
from helper.other import drop_file_type


class MLP(torch.nn.Module):
    def __init__(self, params: dict, nn_weights_path=str()) -> None:
        super().__init__()
        self.in_dim = int(params["in_dim"])
        self.out_dim = int(params["out_dim"])
        self.hid_dim = int(params["hid_dim"])
        self.num_hid_layer = int(params["num_hid_layer"])
        self.act_type: str = params["act_type"]
        if self.act_type == "relu":
            self.actFun = torch.nn.ReLU()
        else:
            raise NotImplementedError(f"activation {self.act_type} is not supported")

        tmp = [self.in_dim] + [self.hid_dim] * (self.num_hid_layer + 1) + [self.out_dim]
        mlp = torch.nn.ModuleList()
        for i in range(len(tmp) - 2):
            mlp.append(torch.nn.Linear(tmp[i], tmp[i + 1]))
            mlp.append(self.actFun)
        mlp.append(torch.nn.Linear(tmp[-2], tmp[-1]))
        self.mlp = mlp
        if nn_weights_path:
            assert os.path.exists(nn_weights_path), f"{nn_weights_path} doesnt exist"
            self.load_state_dict(torch.load(nn_weights_path))
            print("model loaded!")
        else:
            print("model initilized!")

    def set_weights(self, np_seed:int=None, method=str()):
        """
            the method is not consistent with the litrature
            it is used for checking with tf implemetation
        """
        if np_seed is not None:
            np.random.seed(np_seed)
        for i, layer in enumerate(self.mlp):
            if (i+1)%2 == 0:
                continue
            with torch.no_grad():
                w, b = layer.weight.data, layer.bias.data
                ww = np.random.randn(w.shape[0], w.shape[1]) / np.sqrt(max(w.shape))
                bb = np.random.randn(b.shape[0]) / np.sqrt(b.shape[0])
                
                layer.weight.data = torch.FloatTensor(ww)
                layer.bias.data =  torch.FloatTensor(bb)


    def forward(self, x: torch.tensor) -> torch.tensor:
        y = x
        for f in self.mlp:
            y = f(y)
        return y

    def save(self, dir_to_save: str, model_info_name: str, weight_name: str) -> None:
        weight_path = os.path.join(dir_to_save, weight_name)
        torch.save(self.state_dict(), weight_path)
        tmp = {"weight_path": weight_path}
        for k, v in self.__dict__.items():
            if k in {"in_dim", "out_dim", "hid_dim", "num_hid_layer", "act_type"}:
                tmp[k] = v
        tmp["torch_version"] = torch.__version__
        tmp["model"] = "MLP"
        tmp_name_ = drop_file_type(model_info_name, "json")
        with open(os.path.join(dir_to_save, tmp_name_ + ".json"), "w") as f:
            json.dump(tmp, f)
        print("model saved!")
