# src/ml/models/graphsage.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, road_features, working_space, road_types, dropout):
        super().__init__()
        # road_features: The number of features each road segment has.
        # working_space: The hidden size. The first layer describes each segment with this many learned numbers,
        #                which the second layer uses to score the road types.
        
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
        return self.layer2(x, edge_index)
