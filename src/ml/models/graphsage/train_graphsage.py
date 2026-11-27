# src/ml/models/graphsage/train_graphsage.py

# Prepares the data and trains GraphSAGE. Used by dev_run_graphsage.py (scored on the validation set during
# development) and final_testrun_graphsage.py (scored on the test set once, at the end).

import copy

import torch
import torch.nn.functional as F

from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.database_loader import load_segments
from src.ml.data.road_graph_builder import RoadGraphBuilder, ROAD_CLASSES, neighbour_label_shares
from src.ml.data.split_graph import make_split_masks
from src.ml.device import describe_device, get_device, graph_to_device
from src.ml.evaluation import evaluate, labels_in_data
from src.ml.models.graphsage import graphsage_config as sage_config
from src.ml.models.graphsage.graphsage_model import GraphSAGE


def prepare_data():
    # Everything that is the same for every training run: the graph, the split, the standardised features, the class
    # weights and the road types that occur. Done once, then reused for every seed.
    # Returns graph, train_mask, val_mask, test_mask, class_weights, labels.

    # Load the Beijing road network and the labels from the database, and build the line graph
    # (one vertex per road segment)
    segments = load_segments()

    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()

    # The labels come from the database (src/db/create_road_type_labels.py). Link roads get the label of their
    # parent type. These segments have no label:
    # - Segments with several road types at once (e.g. "['residential', 'tertiary']" or
    #   "['motorway_link', 'primary_link']"), because OSMnx merged road pieces of different types into one edge.
    # - Segments of a rare road type, or any other type that is not in ROAD_CLASSES.
    # The graph builder gives these segments y = -1 and mask = False: they stay in the graph as neighbours, but are
    # never used for training or evaluation.
    print("Segments:", len(segments), "Labelled:", graph["mask"].sum().item())

    # Split the labelled segments into train, validation and test. Strategy, seed and shares come from the config, so
    # every model uses the same split.
    masks = make_split_masks(graph)
    train_mask, val_mask, test_mask = masks["train"], masks["validation"], masks["test"]
    print("Train:", train_mask.sum().item(), "Validation:", val_mask.sum().item(), "Test:", test_mask.sum().item())

    # Standardise the features, with the mean and standard deviation of the training segments only
    builder.standardize_features(segments, train_mask)

    # The neighbour label columns are not added here: train_graphsage() makes them itself, with group hiding

    # Weights for the loss, so mistakes on rare road types cost more than mistakes on common ones. Without them the
    # model mostly learns residential, which is about half of the segments. Counted from the training segments only.
    class_weights = compute_class_weights(graph["y"][train_mask], len(ROAD_CLASSES), config.CLASS_WEIGHT_POWER)

    # The road types that occur in the data. Macro-F1 is averaged over these, so road types without any segments do not pull it down.
    labels = labels_in_data(graph["y"], graph["mask"])

    return graph, train_mask, val_mask, test_mask, class_weights, labels


def saved_settings():
    # The settings saved with every GraphSAGE run in the database (model_run.hyperparameters), so a run can be
    # trained again with the same settings
    return {
        "epochs": config.EPOCHS,
        "hidden": sage_config.WORKING_SPACE,
        "dropout": sage_config.DROPOUT,
        "learning_rate": sage_config.LEARNING_RATE,
        "weight_decay": sage_config.WEIGHT_DECAY,
        "groups": sage_config.GROUPS,
        "validate_every": sage_config.VALIDATE_EVERY,
        "class_weight_power": config.CLASS_WEIGHT_POWER,
    }


def train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed, history=None):
    # Trains one GraphSAGE model with the given seed and returns its predictions for every segment, from the epoch with
    # the best validation macro-F1. Returns predictions, best_epoch.
    # The test set is not passed in, so it can never influence training or the choice of epoch.
    # If history is a list, one row per epoch is added to it: {"epoch", "train_loss", "val_macro_f1"}. The run scripts
    # save it in the database as the training curve. val_macro_f1 is None for the epochs that were not validated.

    # The second layer outputs one class score for each road type, read from the class list so they stay in sync.
    road_types = len(ROAD_CLASSES)

    # With 1 group every training label would be hidden, so the neighbour label columns would always be 0
    if sage_config.GROUPS < 2:
        raise ValueError("GROUPS in graphsage_config.py must be at least 2")

    # Train on the GPU if there is one, otherwise on the CPU (see src/ml/device.py). The model and every tensor it
    # touches must be on the same device, so the graph, the masks and the class weights are moved there too.
    # The training segment numbers stay on the CPU, because the group shuffling below uses a CPU random generator.
    device = get_device()
    train_indices = train_mask.nonzero().view(-1)
    graph = graph_to_device(graph, device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    class_weights = class_weights.to(device)

    # The input is the normal features plus one column per road type: the share of the neighbours that are training
    # segments of that road type (see neighbour_label_shares). For validation and the predictions we use ALL the
    # training labels, which is the real situation: every known label is used to predict the missing ones.
    full_x = torch.cat([graph["x"], neighbour_label_shares(graph["edge_index"], graph["y"], train_mask, road_types)], dim=1)

    # The number of features each road segment has, read from the data so it stays correct if features are added.
    road_features = full_x.shape[1]

    # Fix the random starting weights and dropout pattern, so every run with this seed gives the same result. This must
    # come before the model is created, because that is when the starting weights are drawn.
    torch.manual_seed(seed)

    # The random groups for group hiding. A separate generator with the same seed, so the groups are the same every
    # time we train with this seed.
    generator = torch.Generator().manual_seed(seed)

    # Hidden size, dropout, learning rate and weight decay come from the GraphSAGE config
    # (src/ml/models/graphsage/graphsage_config.py)
    model = GraphSAGE(road_features, sage_config.WORKING_SPACE, road_types, sage_config.DROPOUT).to(device)
    # Adam changes the model's weights to lower the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=sage_config.LEARNING_RATE,
                                 weight_decay=sage_config.WEIGHT_DECAY)

    # Show what is being trained and with which settings
    print(f"Training GraphSAGE with seed {seed} on {describe_device(device)}")
    print(f"GraphSAGE with {road_features} features, {sage_config.WORKING_SPACE} hidden units, dropout {sage_config.DROPOUT}, learning rate {sage_config.LEARNING_RATE}, weight decay {sage_config.WEIGHT_DECAY}, {sage_config.GROUPS} groups")

    # The best validation macro-F1 so far, the epoch it came from and a copy of the model's weights at that epoch.
    # best_f1 starts at -1, so the first epoch (with a score of 0 or more) is always saved.
    best_f1 = -1
    best_epoch = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        # Training mode turns dropout on again, after the validation check of the previous epoch turned it off
        model.train()

        # Group hiding: shuffle the training segments and split them into GROUPS groups of (almost) the same size.
        # randperm gives the numbers 0 .. n-1 in a random order, and chunk cuts the list into GROUPS pieces.
        shuffled = train_indices[torch.randperm(len(train_indices), generator=generator)]
        groups = [group.to(device) for group in shuffled.chunk(sage_config.GROUPS)]

        loss_sum = 0.0
        for group in groups:
            # Hide the labels of this group: the neighbour label columns are built from the other training labels only.
            # So the segments in the group don't know their own label, not even through their neighbours, just like
            # a segment whose label is really missing.
            known = train_mask.clone()
            known[group] = False
            x = torch.cat([graph["x"], neighbour_label_shares(graph["edge_index"], graph["y"], known, road_types)], dim=1)

            # Clear the gradients from the previous step. PyTorch adds new gradients to the old ones, so without this
            # every step would also include the directions of all earlier steps.
            optimizer.zero_grad()

            # Forward: one row of scores per segment, one score per road type
            out = model(x, graph["edge_index"])
            # Loss: how wrong the scores are, measured on the hidden group only (validation and test labels are never
            # used for learning). Cross-entropy is low when the correct road type gets a high score compared to the
            # others. The class weights make mistakes on rare road types count more.
            loss = F.cross_entropy(out[group], graph["y"][group], weight=class_weights)
            # Backward: work out for every weight which direction lowers the loss (the gradients)
            loss.backward()
            # Step: move every weight a little in that direction
            optimizer.step()
            loss_sum = loss_sum + loss.item()

        # The average loss over the groups, only for printing
        loss = loss_sum / len(groups)

        # Only check the validation set every VALIDATE_EVERY epochs (and after the last epoch), since the check runs the
        # model on the whole graph a second time. The best epoch is then chosen from the checked epochs only.
        if (epoch + 1) % sage_config.VALIDATE_EVERY != 0 and epoch + 1 != config.EPOCHS:
            print(f"Epoch {epoch + 1}: training loss {loss:.4f}")
            if history is not None:
                history.append({"epoch": epoch + 1, "train_loss": loss, "val_macro_f1": None})
            continue

        # Check the model on the validation set after this epoch. Evaluation mode turns dropout off, and gradients are
        # only needed for training, so they are switched off to save memory and time.
        model.eval()
        with torch.no_grad():
            out = model(full_x, graph["edge_index"])
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
            history.append({"epoch": epoch + 1, "train_loss": loss, "val_macro_f1": val_f1})

        print(f"Epoch {epoch + 1}: training loss {loss:.4f}, validation macro-F1 {val_f1:.4f}{' (best so far)' if new_best else ''}")
        print(f"Best epoch so far: {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Go back to the weights from the best epoch, so the predictions come from the best model and not the last one
    model.load_state_dict(best_state)
    print(f"Using the model from epoch {best_epoch} (validation macro-F1 {best_f1:.4f})")

    # Predict again with the best weights. The predictions from the loop came from the last epoch's weights, and
    # load_state_dict only changes the model, not predictions that were already made. If the best epoch was not the
    # last one, those predictions would belong to a different model than the one reported.
    # The model is already in evaluation mode from the loop, but setting it again keeps this part correct on its own.
    model.eval()
    with torch.no_grad():
        out = model(full_x, graph["edge_index"])
    # The road type with the highest score is the model's prediction. Moved back to the CPU, so the callers can index
    # it with their own (CPU) masks and score it with scikit-learn.
    predictions = out.argmax(dim=1).cpu()

    return predictions, best_epoch
