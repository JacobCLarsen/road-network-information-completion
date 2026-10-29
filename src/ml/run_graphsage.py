# src/ml/run_graphsage.py

import torch
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES
from src.ml.models.graphsage import GraphSAGE


def make_split(mask, seed, train_share=0.6, val_share=0.2):
    """
    TEMPORARY random split, to be replaced by the group's split later. Anything that returns the same three masks
    can take its place.

    Only labelled segments (mask = True) are split. Returns train_mask, val_mask and test_mask: one True/False value
    per segment, with no segment in more than one mask. The same seed always gives the same split.
    """
    # Positions of the labelled segments, in random order
    labelled = mask.nonzero().view(-1)
    generator = torch.Generator().manual_seed(seed)
    shuffled = labelled[torch.randperm(len(labelled), generator=generator)]

    # Cut the shuffled list into train, validation and test (the rest)
    n_train = int(train_share * len(shuffled))
    n_val = int(val_share * len(shuffled))
    parts = [shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]]

    # Turn each part into a True/False mask over all segments
    masks = []
    for part in parts:
        part_mask = torch.zeros(len(mask), dtype=torch.bool)
        part_mask[part] = True
        masks.append(part_mask)

    train_mask, val_mask, test_mask = masks
    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    # Load the Beijing road network and build the line graph (one vertex per road segment)
    loader = ShapefileMapLoader("data/bj_data/map")
    segments = loader.get_segments()

    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()

    # The number of features each road segment has, read from the data so it stays correct if features are added.
    road_features = graph["x"].shape[1]
    # Hidden size: after the first layer, each road segment is described by this many learned numbers, which the
    # second layer uses to score the road types. Bigger can capture more patterns but is slower and overfits more
    # easily. 64 is a starting point; tune it on the validation set.
    working_space = 64
    # The second layer outputs one class score for each road type, read from the class list so they stay in sync.
    road_types = len(ROAD_CLASSES)
    # Dropout is only active during training. It sets some values to zero so the model learns patterns instead of
    # memorising the training roads.
    dropout = 0.5

    model = GraphSAGE(road_features, working_space, road_types, dropout)
    out = model(graph["x"], graph["edge_index"])

    print("Segments:", len(segments))
    print("x:", graph["x"].shape)
    print("edge_index:", graph["edge_index"].shape)
    print("Model output:", out.shape)
    # Every road has a highway tag: in OpenStreetMap, the highway tag is what marks a line as a road, and the map was
    # made with OSMnx (the u, v, key and osmid fields are its format), which only downloads lines that have one.
    # But 912 segments have a tag that is not in ROAD_CLASSES:
    # - 906 have several road types at once (e.g. "['residential', 'tertiary']" or "['motorway_link', 'primary_link']"),
    #   because OSMnx merged road pieces of different types into one edge.
    # - 6 have a single type that is not in the class list (4 "disused" and 2 "rest_area").
    # The graph builder gives these segments y = -1 and mask = False: they stay in the graph as neighbours, but are
    # never used for training or evaluation.
    print("Labeled:", graph["mask"].sum().item())
    print("Unlabeled:", (graph["y"] == -1).sum().item())

    train_mask, val_mask, test_mask = make_split(graph["mask"], seed=42)
    print("Train:", train_mask.sum().item(), "Validation:", val_mask.sum().item(), "Test:", test_mask.sum().item())
