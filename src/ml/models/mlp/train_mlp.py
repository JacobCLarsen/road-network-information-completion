# src/ml/models/mlp/train_mlp.py

# Prepares the data and trains the MLP. Used by dev_run_mlp.py (scored on the validation set during development) and
# final_testrun_mlp.py (scored on the test set once, at the end).

"""Imports"""
import copy
import torch
import torch.nn.functional as F
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.database_loader import load_segments
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES
from src.ml.data.split_graph import make_split_masks
from src.ml.device import describe_device, get_device, graph_to_device
from src.ml.evaluation import evaluate, labels_in_data
from src.ml.models.mlp import mlp_config
from src.ml.models.mlp.mlp_model import MLP

def prepare_data():
    # Everything that is the same for every training run: the graph, the split, the standardised features, the class
    # weights and the road types that occur. Done once, then reused for every seed.
    # Returns graph, train_mask, val_mask, test_mask, class_weights, labels.
    #
    # The MLP does not use the connections between segments, but the graph builder is still used, so the MLP gets
    # exactly the same features, labels and split as GraphSAGE and the other models.

    # Load the Beijing road network and the labels from the database, and build the line graph
    # (one vertex per road segment)
    segments = load_segments()

    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()

    # Segments without a usable road type have y = -1 and mask = False. They are never used for training or evaluation
    # (see train_graphsage.py for which segments these are).
    print("Segments:", len(segments), "Labelled:", graph["mask"].sum().item())

    # Split the labelled segments into train, validation and test. Strategy, seed and shares come from the config, so
    # every model uses the same split.
    masks = make_split_masks(graph)
    train_mask, val_mask, test_mask = masks["train"], masks["validation"], masks["test"]
    print("Train:", train_mask.sum().item(), "Validation:", val_mask.sum().item(), "Test:", test_mask.sum().item())

    # Standardise the features, with the mean and standard deviation of the training segments only
    builder.standardize_features(segments, train_mask)

    # Add the share of each road type among the neighbours (training labels only). x gets new columns,
    # so the graph dictionary needs the new x
    builder.add_neighbour_label_features(train_mask)
    graph["x"] = builder.x

    # Weights for the loss, so mistakes on rare road types cost more than mistakes on common ones. Counted from the
    # training segments only. The same shared function as GraphSAGE, so both are trained with the same weights.
    class_weights = compute_class_weights(graph["y"][train_mask], len(ROAD_CLASSES), config.CLASS_WEIGHT_POWER)

    # The road types that occur in the data. Macro-F1 is averaged over these, so road types without any segments do not pull it down.
    labels = labels_in_data(graph["y"], graph["mask"])

    return graph, train_mask, val_mask, test_mask, class_weights, labels

def saved_settings():
    # The settings saved with every MLP run in the database (model_run.hyperparameters), so a run can be trained again
    # with the same settings
    return {
        "epochs": config.EPOCHS,
        "hidden": mlp_config.WORKING_SPACE,
        "dropout": mlp_config.DROPOUT,
        "learning_rate": mlp_config.LEARNING_RATE,
        "class_weight_power": config.CLASS_WEIGHT_POWER,
    }

def train_mlp(graph, train_mask, val_mask, class_weights, labels, seed, history=None):
    # Trains one MLP with the given seed and returns its predictions for every segment, from the epoch with the best
    # validation macro-F1. Returns predictions, best_epoch.
    # The test set is not passed in, so it can never influence training or the choice of epoch.
    # If history is a list, one row per epoch is added to it: {"epoch", "train_loss", "val_macro_f1"}. The run scripts
    # save it in the database as the training curve.

    # Train on the GPU if there is one, otherwise on the CPU (see src/ml/device.py). The model and every tensor it
    # touches must be on the same device, so the graph, the masks and the class weights are moved there too.
    device = get_device()
    graph = graph_to_device(graph, device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    class_weights = class_weights.to(device)

    # The number of features each road segment has, read from the data so it stays correct if features are added.
    road_features = graph["x"].shape[1]
    # The second layer outputs one class score for each road type, read from the class list so they stay in sync.
    road_types = len(ROAD_CLASSES)

    # Fix the random starting weights and dropout pattern, so every run with this seed gives the same result. This must
    # come before the model is created, because that is when the starting weights are drawn.
    torch.manual_seed(seed)

    # Hidden size, dropout and learning rate come from the MLP config (src/ml/models/mlp/mlp_config.py)
    model = MLP(road_features, mlp_config.WORKING_SPACE, road_types, mlp_config.DROPOUT).to(device)
    # Adam changes the model's weights to lower the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_config.LEARNING_RATE)

    # Show what is being trained and with which settings
    print(f"Training MLP with seed {seed} on {describe_device(device)}")
    print(f"MLP with {road_features} features, {mlp_config.WORKING_SPACE} hidden units, dropout {mlp_config.DROPOUT}, learning rate {mlp_config.LEARNING_RATE}")

    # The best validation macro-F1 so far, the epoch it came from and a copy of the model's weights at that epoch.
    # best_f1 starts at -1, so the first epoch (with a score of 0 or more) is always saved.
    best_f1 = -1
    best_epoch = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        # Training mode turns dropout on again, after the validation check of the previous epoch turned it off
        model.train()

        # Clear the gradients from the previous epoch, so every step only uses the gradients of this epoch
        optimizer.zero_grad()

        # Forward: one row of scores per segment, one score per road type. Like GraphSAGE, every segment is passed in
        # at once (one weight change per epoch), so both models get the same number of training steps.
        out = model(graph["x"])
        # Loss on the training segments only (validation and test labels are never used for learning). The class
        # weights make mistakes on rare road types count more.
        loss = F.cross_entropy(out[train_mask], graph["y"][train_mask], weight=class_weights)
        # Backward: work out for every weight which direction lowers the loss (the gradients)
        loss.backward()
        # Step: move every weight a little in that direction
        optimizer.step()

        # Check the model on the validation set after this epoch. Evaluation mode turns dropout off, and gradients are
        # only needed for training, so they are switched off to save memory and time.
        model.eval()
        with torch.no_grad():
            out = model(graph["x"])
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

        if history is not None:
            history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_macro_f1": val_f1})

        print(f"Epoch {epoch + 1}: training loss {loss.item():.4f}, validation macro-F1 {val_f1:.4f}{' (best so far)' if new_best else ''}")
        print(f"Best epoch so far: {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Go back to the weights from the best epoch, so the predictions come from the best model and not the last one
    model.load_state_dict(best_state)
    print(f"Using the model from epoch {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Predict again with the best weights. The predictions from the loop came from the last epoch's weights.
    model.eval()
    with torch.no_grad():
        out = model(graph["x"])
    # The road type with the highest score is the model's prediction. Moved back to the CPU, so the callers can index
    # it with their own (CPU) masks and score it with scikit-learn.
    predictions = out.argmax(dim=1).cpu()

    return predictions, best_epoch
