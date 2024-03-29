import torch

class ManyToMany(torch.nn.Module):
    def __init__(self, params: dict) -> None:
        super().__init__()
        """
            batch_first is True! -> (b, seq, feat)
        """
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.seq_length = params['seq_length']
        bias = params['bias']
        act_type = params['act_type']
        assert act_type in {"relu", "tanh"}
        self.num_rec_layers = params['num_rec_layers']
        self.rnn = torch.nn.RNN(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_rec_layers,
                                nonlinearity = act_type,
                                bias = bias,
                                batch_first = True,
                                )
        
    
    def forward(self, x: torch.tensor):
        # batch first as default
        assert x.shape[1:] == (self.seq_length, self.input_size)
        bsz = x.shape[0]
        h = torch.zeros(self.num_rec_layers, bsz, self.hidden_size)
        o, h = self.rnn(x, h)
        return o
        

class ManyToOne(ManyToMany):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
    
    def forward(self, x: torch.tensor):
        return super().forward(x)[:, -1, :] # last seq is the representation


class OneToMany(ManyToMany):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
    
    def forward(self, x: torch.tensor):
        if x.dim() == 2:
            x_ = x.reshape(x.shape[0], 1, x.shape[1])
            x_ = x_.repeat([1, self.seq_length, 1])
        elif x.dim() == 3:
            assert x.shape[1] == 1 # one
            x_ = x.repeat([1, self.seq_length, 1])
        else:
            raise NotImplementedError()
        return super().forward(x_)