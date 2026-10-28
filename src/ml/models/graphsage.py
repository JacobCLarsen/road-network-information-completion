# src/ml/models/graphsage.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, road_features, working_space, road_types):
        super().__init__()
        # road_features: The number of features each road segment has.
        # working_space: The first layer outputs a working space of this many features.
        # road_types: The second layer outputs one class score for each road type.
        self.layer1 = SAGEConv(in_channels=road_features, out_channels=working_space)
        self.layer2 = SAGEConv(in_channels=working_space, out_channels=road_types)

    def forward(self, x, edge_index):

        pass

if __name__ == "__main__":
    # The number of features each road segment has. This is currently 4.
    road_features = 4
    # The first layer outputs a working space of 64 features. This is arbitrary and can be changed if needed.
    working_space = 64
    # The second layer outputs 21 class scores, one for each roadtype there is right now. Should be changed if the number of roadtypes changes or new roadtypes are added.
    road_types = 21
    model = GraphSAGE(road_features, working_space, road_types)
    print(model)

