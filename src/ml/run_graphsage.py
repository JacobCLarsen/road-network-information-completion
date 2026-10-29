# src/ml/run_graphsage.py

import torch
import torch.nn.functional as F
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES
from src.ml.evaluation import evaluate, labels_in_data
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

    # Every road has a highway tag: in OpenStreetMap, the highway tag is what marks a line as a road, and the map was
    # made with OSMnx (the u, v, key and osmid fields are its format), which only downloads lines that have one.
    # But 912 segments have a tag that is not in ROAD_CLASSES:
    # - 906 have several road types at once (e.g. "['residential', 'tertiary']" or "['motorway_link', 'primary_link']"),
    #   because OSMnx merged road pieces of different types into one edge.
    # - 6 have a single type that is not in the class list (4 "disused" and 2 "rest_area").
    # The graph builder gives these segments y = -1 and mask = False: they stay in the graph as neighbours, but are
    # never used for training or evaluation.
    print("Segments:", len(segments))
    print("Labeled:", graph["mask"].sum().item())
    print("Unlabeled:", (graph["y"] == -1).sum().item())

    # Split the labelled segments into train, validation and test
    train_mask, val_mask, test_mask = make_split(graph["mask"], seed=42)
    print("Train:", train_mask.sum().item(), "Validation:", val_mask.sum().item(), "Test:", test_mask.sum().item())

    # Standardise every feature to mean 0 and standard deviation 1, so no feature dominates because of its scale.
    # The mean and standard deviation come from the training segments only, so nothing is learned from the
    # validation and test segments. They are then applied to all segments, because the model uses every segment's
    # features as neighbours.
    feature_mean = graph["x"][train_mask].mean(dim=0)
    feature_std = graph["x"][train_mask].std(dim=0)
    x_standardized = (graph["x"] - feature_mean) / feature_std

    # The number of features each road segment has, read from the data so it stays correct if features are added.
    road_features = x_standardized.shape[1]
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
    # Adam changes the model's weights to lower the loss. lr (the learning rate) is how big each change is: too big
    # and training jumps around, too small and it learns very slowly. 0.01 is a common starting point.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train for 50 epochs. Each epoch runs the model on the whole graph once and changes the weights once.
    for epoch in range(50):
        # Clear the gradients from the previous epoch. PyTorch adds new gradients to the old ones, so without this
        # every step would also include the directions of all earlier steps.
        optimizer.zero_grad()

        # Forward: one row of scores per segment, one score per road type
        out = model(x_standardized, graph["edge_index"])
        # Loss: how wrong the scores are, measured on the training segments only (validation and test labels are never
        # used for learning). Cross-entropy is low when the correct road type gets a high score compared to the others.
        # An untrained model starts at around ln(number of road types), the loss of guessing all road types equally.
        loss = F.cross_entropy(out[train_mask], graph["y"][train_mask])
        # Backward: work out for every weight which direction lowers the loss (the gradients)
        loss.backward()
        # Step: move every weight a little in that direction
        optimizer.step()

        print(f"Training loss after epoch {epoch + 1}: {loss.item():.4f}")

    # Evaluation mode turns dropout off, so the trained model gives its full predictions
    model.eval()
    out = model(x_standardized, graph["edge_index"])
    # The road type with the highest score is the model's prediction
    predictions = out.argmax(dim=1)

    # Score the predictions on the validation segments with the shared evaluation function. Macro-F1 is averaged over
    # the road types that occur in the data, so road types without any segments do not pull it down. This list will
    # change once the group has decided the final class set.
    labels = labels_in_data(graph["y"], graph["mask"])
    result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)

    # Macro-F1 is the main metric. Grouped macro-F1 scores only the coarse categories (highways, main roads, minor
    # roads), and accuracy is shown for comparison with the majority-class baseline.
    print(f"Validation macro-F1: {result['macro_f1']:.4f}")
    print(f"Validation grouped macro-F1: {result['grouped_macro_f1']:.4f}")
    print(f"Validation accuracy: {result['accuracy']:.4f}")
    for name, scores in result["per_class"].items():
        print(f"{name}: precision {scores['precision']:.3f}, recall {scores['recall']:.3f}, F1 {scores['f1']:.3f}")
