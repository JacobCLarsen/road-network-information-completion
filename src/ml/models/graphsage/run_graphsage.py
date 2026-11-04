# src/ml/models/graphsage/run_graphsage.py

import torch
import torch.nn.functional as F
import copy

from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES
from src.ml.data.split import train_val_test_split
from src.ml.evaluation import evaluate, labels_in_data
from src.ml.models.graphsage import config as sage_config
from src.ml.models.graphsage.graphsage import GraphSAGE


if __name__ == "__main__":
    # Load the Beijing road network and build the line graph (one vertex per road segment)
    loader = ShapefileMapLoader(config.DATA_DIR)
    segments = loader.get_segments()

    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()

    # Every road has a highway tag: in OpenStreetMap, the highway tag is what marks a line as a road, and the map was
    # made with OSMnx (the u, v, key and osmid fields are its format), which only downloads lines that have one.
    # Link roads get the label of their parent type. These segments have no label:
    # - Segments with several road types at once (e.g. "['residential', 'tertiary']" or
    #   "['motorway_link', 'primary_link']"), because OSMnx merged road pieces of different types into one edge.
    # - Segments of a rare road type (RARE_ROAD_TYPES) or any other type that is not in ROAD_CLASSES.
    # The graph builder gives these segments y = -1 and mask = False: they stay in the graph as neighbours, but are
    # never used for training or evaluation.
    print("Segments:", len(segments), "Labelled:", graph["mask"].sum().item())

    # Split the labelled segments into train, validation and test
    train_mask, val_mask, test_mask = train_val_test_split(graph["mask"], config.SPLIT_SEED, config.TRAIN_SHARE, config.VAL_SHARE)
    print("Train:", train_mask.sum().item(), "Validation:", val_mask.sum().item(), "Test:", test_mask.sum().item())
    
    # Standardise the features, with the mean and standard deviation of the training segments only
    builder.standardize_features(segments, train_mask)

    # The number of features each road segment has, read from the data so it stays correct if features are added.
    road_features = graph["x"].shape[1]
    # The second layer outputs one class score for each road type, read from the class list so they stay in sync.
    road_types = len(ROAD_CLASSES)

    # Weights for the loss, so mistakes on rare road types cost more than mistakes on common ones. Without them the
    # model mostly learns residential, which is about half of the segments. Counted from the training segments only.
    class_weights = compute_class_weights(graph["y"][train_mask], road_types)

    # Fix the random starting weights and dropout pattern, so every run gives the same result. This must come before
    # the model is created, because that is when the starting weights are drawn.
    torch.manual_seed(config.MODEL_SEED)

    # Hidden size, dropout and learning rate come from the GraphSAGE config (src/ml/models/graphsage/config.py)
    model = GraphSAGE(road_features, sage_config.WORKING_SPACE, road_types, sage_config.DROPOUT)
    # Adam changes the model's weights to lower the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=sage_config.LEARNING_RATE)

    # The road types that occur in the data. Macro-F1 is averaged over these, so road types without any segments do not pull it down.
    labels = labels_in_data(graph["y"], graph["mask"])

    # adds newlines between the print statements, so the output is easier to read when the script is run in a terminal
    print("Training GraphSAGE model on the Beijing road network")
    print(f"GraphSAGE with {road_features} features, {sage_config.WORKING_SPACE} hidden units, dropout {sage_config.DROPOUT}, learning rate {sage_config.LEARNING_RATE}")

    # The best validation macro-F1 so far, the epoch it came from and a copy of the model's weights at that epoch.
    # best_f1 starts at -1, so the first epoch (with a score of 0 or more) is always saved.
    best_f1 = -1
    best_epoch = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        # Training mode turns dropout on again, after the validation check of the previous epoch turned it off
        model.train()

        # Clear the gradients from the previous epoch. PyTorch adds new gradients to the old ones, so without this
        # every step would also include the directions of all earlier steps.
        optimizer.zero_grad()

        # Forward: one row of scores per segment, one score per road type
        out = model(graph["x"], graph["edge_index"])
        # Loss: how wrong the scores are, measured on the training segments only (validation and test labels are never
        # used for learning). Cross-entropy is low when the correct road type gets a high score compared to the others.
        # An untrained model starts at around ln(number of road types), the loss of guessing all road types equally.
        # The class weights make mistakes on rare road types count more.
        loss = F.cross_entropy(out[train_mask], graph["y"][train_mask], weight=class_weights)
        # Backward: work out for every weight which direction lowers the loss (the gradients)
        loss.backward()
        # Step: move every weight a little in that direction
        optimizer.step()

        # Check the model on the validation set after this epoch. Evaluation mode turns dropout off, and gradients are
        # only needed for training, so they are switched off to save memory and time.
        model.eval()
        with torch.no_grad():
            out = model(graph["x"], graph["edge_index"])
        # The road type with the highest score is the model's prediction
        predictions = out.argmax(dim=1)
        val_f1 = evaluate(graph["y"][val_mask], predictions[val_mask], labels)["macro_f1"]

        # Keep a copy of the weights whenever validation macro-F1 is the best so far. deepcopy makes a real copy:
        # state_dict() only points to the live weights, which keep changing as training goes on.
        new_best = val_f1 > best_f1
        if new_best:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}: training loss {loss.item():.4f}, validation macro-F1 {val_f1:.4f}{' (best so far)' if new_best else ''}")
        print(f"Best epoch so far: {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Go back to the weights from the best epoch, so the final evaluation uses the best model and not the last one
    model.load_state_dict(best_state)
    print(f"Using the model from epoch {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Predict again with the best weights. The predictions from the loop came from the last epoch's weights, and
    # load_state_dict only changes the model, not predictions that were already made. If the best epoch was not the
    # last one, those predictions would belong to a different model than the one reported.
    # The model is already in evaluation mode from the loop, but setting it again keeps this part correct on its own.
    model.eval()
    with torch.no_grad():
        out = model(graph["x"], graph["edge_index"])
    # The road type with the highest score is the model's prediction
    predictions = out.argmax(dim=1)

    # Score the predictions on the validation segments with the shared evaluation function
    result = evaluate(graph["y"][val_mask], predictions[val_mask], labels, class_names=ROAD_CLASSES)

    # Macro-F1 is the main metric. Grouped macro-F1 scores only the coarse categories (highways, main roads, minor roads),
    # and accuracy is shown for comparison with the majority-class baseline.
    print(f"Validation macro-F1: {result['macro_f1']:.4f}")
    print(f"Validation grouped macro-F1: {result['grouped_macro_f1']:.4f}")
    print(f"Validation accuracy: {result['accuracy']:.4f}")
    for name, scores in result["per_class"].items():
        print(f"{name}: precision {scores['precision']:.3f}, recall {scores['recall']:.3f}, F1 {scores['f1']:.3f}")
