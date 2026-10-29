import pytest
import torch
from src.ml.models.graphsage.graphsage import GraphSAGE


# A small hand-made graph of three road segments in a row: 0 - 1 - 2.
# Each row of x is one segment's features: log(length), curvature, one-way, degree.
@pytest.fixture
def small_graph():
    x = torch.tensor([[4.2, 0.1, 1.0, 1.0],
                      [5.0, 0.0, 0.0, 2.0],
                      [3.8, 0.3, 0.0, 1.0]])

    # Every connection is stored in both directions: 0 <-> 1 and 1 <-> 2
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]])

    return x, edge_index


def test_output_shape(small_graph):
    x, edge_index = small_graph

    road_features = 4
    working_space = 64
    road_types = 21
    dropout = 0.5

    model = GraphSAGE(road_features, working_space, road_types, dropout)
    out = model(x, edge_index)

    # One row per road segment, one score per road type
    assert out.shape == (x.shape[0], road_types)
