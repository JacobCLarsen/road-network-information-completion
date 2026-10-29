# src/ml/models/graphsage.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, road_features, working_space, road_types, dropout):
        super().__init__()
        # road_features: The number of features each road segment has.
        # working_space: The first layer outputs a working space of this many features.
        # road_types: The second layer outputs one class score for each road type.
        # dropout: The share of values set to zero during training (e.g. 0.5 = half).
        self.layer1 = SAGEConv(in_channels=road_features, out_channels=working_space)
        self.layer2 = SAGEConv(in_channels=working_space, out_channels=road_types)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: The features of every road segment from the graph builder (one row per segment).
        # edge_index: The connected pairs of segments (2 rows: "from" and "to").
        # Returns one row of class scores per segment.
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        # self.training comes from torch.nn.Module: model.train() sets it to True (dropout on),
        # model.eval() sets it to False (dropout off). Always call model.eval() before validation and testing.
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layer2(x, edge_index,)
        

if __name__ == "__main__":
    # The number of features each road segment has. This is currently 4.
    road_features = 4
    # The first layer outputs a working space of 64 features. This is arbitrary and can be changed if needed.
    working_space = 64
    # The second layer outputs 21 class scores, one for each roadtype there is right now. Should be changed if the number of roadtypes changes or new roadtypes are added.
    road_types = 21
    # Dropout is for training, and should only be active during training. It sets some values to zero so we actually learn.
    dropout = 0.5

    model = GraphSAGE(road_features, working_space, road_types, dropout)
    print(model)
    
    x = torch.tensor([[4.2, 0.1, 1.0, 1.0],
                      [5.0, 0.0, 0.0, 2.0],
                      [3.8, 0.3, 0.0, 1.0]])
    
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]])
    
    out = model(x, edge_index)

    print(out.shape)  # Should print torch.Size([3, 21]) since there are 3 segments and 21 road types.
