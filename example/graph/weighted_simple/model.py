import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

class MyMessagePassing(MessagePassing):
    def __init__(self,
        nodeDim,
        edgeDim, 
        outDim, 
        hiddenDim, 
        aggr = 'add',
        act = 'relu'
        ) -> None:
        super().__init__(aggr=aggr)
        self._hiddenDim = hiddenDim
        if act == 'relu':
            actFun =nn.ReLU()
        
        self.edge_net = nn.Sequential(
            nn.Linear(nodeDim*2 + edgeDim, hiddenDim),
            actFun,
            nn.Linear(hiddenDim, hiddenDim),
            actFun,
            nn.LayerNorm(hiddenDim) # maybe before act works better
        )
        self.node_net = nn.Sequential(
            nn.Linear(nodeDim + hiddenDim, hiddenDim),
            actFun,
            nn.Linear(hiddenDim, outDim),
            actFun,
            nn.LayerNorm(outDim)
        )

    def forward(self, graph:Data):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        x_new = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        # x_new += x # residual structure
        
        row, col = edge_index
        edge_attr_new = self.edge_net(torch.cat([x[row], x[col], edge_attr], dim=-1))
        # edge_attr_new += edge_attr
                
        return Data(edge_index=edge_index, x= x_new, edge_attr = edge_attr_new)
    
    def message(self, x_i, x_j, edge_attr):
        """
            x_i: (n, fn)
            x_j: (n, fn)
            edge_attr: (n, fe)
        """
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(x)
    
    def update(self, aggr_out, x):
        """
            aggr_out: (n, c)
        """
        tmp = torch.cat([aggr_out, x], dim=-1)
        return self.node_net(tmp)

class MyNetworkMapper(torch.nn.Module):
    def __init__(self,
        nodeFeatIn,
        edgeFeatIn,
        nodeFeatOut,
        hiddenDim,
        numMsgPass,
        act = 'relu'
        ) -> None:
        super().__init__()
        self.numMsgPass = numMsgPass
        
        if act == 'relu':
            actFun = nn.ReLU()

        # we make edge and node features the same dim as hidden dim
        self.edgeMLP0 = nn.Sequential(
            nn.Linear(edgeFeatIn, hiddenDim),
            actFun,
            nn.Linear(hiddenDim, hiddenDim),
            actFun,
            nn.LayerNorm(hiddenDim)
        )
        self.nodeMLP0 = nn.Sequential(
            nn.Linear(nodeFeatIn, hiddenDim),
            actFun,
            nn.Linear(hiddenDim, hiddenDim),
            actFun,
            nn.LayerNorm(hiddenDim)
        )
        self.nodeMLPFinal = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim),
            actFun,
            nn.Linear(hiddenDim, nodeFeatOut),
        )
        
        # Processor
        self.gnn = MyMessagePassing(
            nodeDim = hiddenDim,
            edgeDim = hiddenDim, 
            outDim = hiddenDim, 
            hiddenDim = hiddenDim,
            )
    
    def forward(self, gData:Data):
        x, edge_index, edge_attr = gData.x, gData.edge_index, gData.edge_attr
        x_ = self.nodeMLP0(x)
        edge_attr_ = self.edgeMLP0(edge_attr)
        graph_ = Data(x=x_, edge_index=edge_index, edge_attr= edge_attr_)
        for _ in range(self.numMsgPass): # num of hoops
             graph_ = self.gnn(graph_)
        
        return self.nodeMLPFinal(graph_.x)
  