# src/ml/models/mlp/mlp_model.py

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    # A multilayer perceptron: it scores the road type of each segment from that segment's own features only, without
    # looking at the connected segments. It has the same two-layer shape as GraphSAGE (graphsage_model.py), so comparing
    # the two shows how much the neighbours in the graph help.
    def __init__(self, road_features, working_space, road_types, dropout):
        super().__init__()
        # road_features: The number of features each road segment has.
        # working_space: The hidden size. The first layer describes each segment with this many learned numbers,
        #                which the second layer uses to score the road types.
        # road_types: The second layer outputs one class score for each road type.
        # dropout: The share of values set to zero during training (e.g. 0.5 = half).
        self.layer1 = torch.nn.Linear(in_features=road_features, out_features=working_space)
        self.layer2 = torch.nn.Linear(in_features=working_space, out_features=road_types)
        self.dropout = dropout

    def forward(self, x):
        # x: The features of the road segments from the graph builder (one row per segment).
        # Returns one row of class scores per segment.
        x = self.layer1(x)
        x = F.relu(x)
        # self.training comes from torch.nn.Module: model.train() sets it to True (dropout on),
        # model.eval() sets it to False (dropout off). Always call model.eval() before validation and testing.
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layer2(x)
