import torch
from torch_geometric.data import Data
from model import MyMessagePassing, MyNetworkMapper

def testMMPBlock():
    numNodes = 4
    nodeFeat = 4
    edgeFeat = 3
    edge_index = [[0, 1], [1, 2], [2, 3], [0, 3]]
    for i, j in edge_index.copy():
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    x = torch.rand(numNodes, nodeFeat)
    edge_attr = torch.rand(edge_index.shape[1], edgeFeat)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(data.x.shape)
    msg = MyMessagePassing(nodeDim=nodeFeat, edgeDim=edgeFeat, outDim=7, hiddenDim=11)
    data = msg(data)
    print(data.x.shape)

def testMyNet():
    numNodes = 4
    nodeFeatIn, nodeFeatOut = 4, 2
    edgeFeatIn = 3
    edge_index = [[0, 1], [1, 2], [2, 3], [0, 3]]
    for i, j in edge_index.copy():
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    x = torch.rand(numNodes, nodeFeatIn)
    y = torch.rand(numNodes, nodeFeatOut)
    edge_attr = torch.rand(edge_index.shape[1], edgeFeatIn)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    print(data.x.shape)
    model = MyNetworkMapper(
        nodeFeatIn=nodeFeatIn,
        edgeFeatIn=edgeFeatIn,
        nodeFeatOut=nodeFeatOut,
        hiddenDim=20,
        numMsgPass=2
    )
    print(model(data))

if __name__ == "__main__":
    # testMMPBlock()
    testMyNet()