# tests/ml/models/mlp/test_train_mlp.py

"""Test MLP training using artificial road segments instead of the full Beijing dataset."""

"""Imports"""
import pytest
import torch
import torch.nn.functional as F
import src.ml.models.mlp.train_mlp as train_module
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import labels_in_data
from src.ml.models.mlp import mlp_config
from src.ml.models.mlp.train_mlp import train_mlp

# The tiny test network is missing some road types, so evaluate() warns about it. That warning is tested in
# tests/ml/test_evaluation.py, so it is hidden here.
pytestmark = pytest.mark.filterwarnings("ignore:No segments of road type:UserWarning")

"""Create a small road network with known features, labels, and data partitions."""
@pytest.fixture
def small_road_network():
    # pytest.fixture makes this network available to tests that include small_road_network as an argument.
    # We use the same artificial network as the GraphSAGE training tests for a consistent comparison.

    # The network contains 12 road segments arranged into two separate roads:
    #
    # 0 - 1 - 2 - 3 - 4 - 10    Road type 0, feature 1.0
    # 5 - 6 - 7 - 8 - 9 - 11    Road type 1, feature -1.0
    #
    # Segment 10 belongs to the test set, while segment 11 has no usable label.
    # The simple feature values make the road types easy for a correctly trained MLP to learn.

    # x contains one artificial feature per segment, while y contains its road-type label.
    # Road type -1 indicates an unlabeled segment that must not be used for training.
    x = torch.tensor([
        [1.0], [1.0], [1.0], [1.0], [1.0], 
        [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], 
        [1.0], [0.0],
    ])

    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1])

    # edge_index describes connections between road segments.
    # The first row contains source segments, and the second contains destination segments.
    # Every connection is stored in both directions because the graph is undirected.
    # The MLP ignores these connections, but GraphSAGE uses them.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 4, 10, 9, 11],
        [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 9, 8, 10, 4, 11, 9],
    ])

    # mask indicates which segments have usable road-type labels.
    # Segment 11 is excluded because its label is -1.
    mask = torch.tensor([
        True, True, True, True, True, True,
        True, True, True, True, True, False,
    ])

    # Combine the features, connections, labels, and usable-label mask into one graph.
    graph = {"x": x, "edge_index": edge_index, "y": y, "mask": mask}

    # Divide the labeled segments into fixed partitions:
    # Training: 0-2 and 5-7, validation: 3-4 and 8-9, and testing: segment 10.
    # Segment 11 is unlabeled and excluded from all partitions.
    # True means the segment belongs to the corresponding partition.
    train_mask = torch.tensor([
        True, True, True, False, False, True,
        True, True, False, False, False, False,
    ])

    val_mask = torch.tensor([
        False, False, False, True, True, False,
        False, False, True, True, False, False,
    ])

    # Calculate class weights using only the training labels.
    # labels_in_data() identifies the road types with usable labels in the graph.
    class_weights = compute_class_weights(y[train_mask], len(ROAD_CLASSES))
    labels = labels_in_data(y, mask)

    return graph, train_mask, val_mask, class_weights, labels

"""Reduce the number of training epochs to keep most tests fast."""
@pytest.fixture
def few_epochs(monkeypatch):
    # monkeypatch temporarily changes the shared EPOCHS setting from config.py.
    # Tests that include few_epochs as an argument train for only five epochs.
    # pytest automatically restores the original setting after each test.
    monkeypatch.setattr(config, "EPOCHS", 5)

"""Check that training returns one prediction for every road segment."""
def test_one_prediction_per_segment(small_road_network, few_epochs):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train the MLP for five epochs using the artificial road network.
    # train_mlp() returns predictions for all segments and the selected best epoch.
    # The underscore ignores the best epoch because it is not needed in this test.
    predictions, _ = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed = 0)

    # Verify that the model returns one prediction for each of the 12 segments.
    # This includes the test segment and the unlabeled segment, even though neither is used for training.
    assert predictions.shape == (12,)

"""Check that training with the same random seed produces identical results."""
def test_same_seed_gives_same_predictions(small_road_network, few_epochs):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train two separate MLPs using exactly the same data, settings, and seed.
    # The seed controls random operations such as model initialization and dropout.
    first_predictions, first_epoch = train_mlp(
        graph, train_mask, val_mask, class_weights, labels, seed = 0
    )

    second_predictions, second_epoch = train_mlp(
        graph, train_mask, val_mask, class_weights, labels, seed = 0
    )

    # torch.equal() checks that both prediction tensors contain exactly the same values.
    # Both training runs must also select the same best epoch.
    assert torch.equal(first_predictions, second_predictions)
    assert first_epoch == second_epoch

"""Check that the MLP learns the relationship between features and road types."""
def test_predicts_all_validation_segments_correctly(small_road_network, monkeypatch):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Five epochs are not sufficient for the model to reliably learn the pattern.
    # Use 50 epochs so the MLP has enough opportunities to update its weights.
    monkeypatch.setattr(config, "EPOCHS", 50)

    # Train the model using only training labels and evaluate its predictions on validation segments.
    predictions, _ = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed = 0)

    # The training segments teach the model that feature 1.0 represents road type 0
    # and feature -1.0 represents road type 1.
    # Verify that it correctly predicts the validation segments it did not train on.
    assert torch.equal(predictions[val_mask], graph["y"][val_mask])

"""Check that road-network connections do not influence MLP training."""
def test_connections_do_not_change_training(small_road_network, few_epochs):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train the MLP using the original graph containing road-network connections.
    with_connections, with_epoch = train_mlp(
        graph, train_mask, val_mask, class_weights, labels, seed = 0
    )

    # Create a copy of the graph where all connections have been removed.
    # torch.empty((2, 0)) creates an edge_index with two rows and zero connections.
    # {**graph, ...} copies the dictionary while replacing only edge_index.
    no_connections = {
        **graph,
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }

    # Train again using identical settings and seed, but without any connections.
    without_connections, without_epoch = train_mlp(
        no_connections, train_mask, val_mask, class_weights, labels, seed = 0
    )

    # Since the MLP does not use neighboring segments, removing connections
    # must not change its predictions or the selected best epoch.
    # This verifies that the MLP remains a valid baseline for comparison with GraphSAGE.
    assert torch.equal(with_connections, without_connections)
    assert with_epoch == without_epoch

"""Record the labels, class weights, and validation scores used during training."""
@pytest.fixture
def training_spies(monkeypatch):
    # A spy records information passed to a function while still allowing the real function to execute.
    # We use spies because train_mlp() only returns predictions and the best epoch,
    # which does not reveal which labels or class weights were used internally.

    # Create lists that will be filled as the MLP trains:
    # loss_labels: labels used to calculate training loss.
    # loss_weights: class weights passed to the loss function in each epoch.
    # evaluate_labels: labels used during validation.
    # val_f1s: validation macro-F1 calculated after each epoch.
    recorded = {
        "loss_labels": [],
        "loss_weights": [],
        "evaluate_labels": [],
        "val_f1s": [],
    }

    # Save the original cross-entropy function before temporarily replacing it.
    # F.cross_entropy() calculates the classification loss used to update the model.
    real_cross_entropy = F.cross_entropy

    def spy_cross_entropy(scores, true_labels, **kwargs):
        # tolist() converts the label tensor into a regular Python list.
        # extend() adds those labels to our recorded list.
        # **kwargs collects additional keyword arguments, including the class weights.
        recorded["loss_labels"].extend(true_labels.tolist())
        recorded["loss_weights"].append(kwargs.get("weight"))

        # Call the original loss function so training continues normally.
        return real_cross_entropy(scores, true_labels, **kwargs)

    # Save the original evaluation function before replacing it with another spy.
    real_evaluate = train_module.evaluate

    def spy_evaluate(true_labels, predictions, *args, **kwargs):
        # Record the labels used during validation.
        # *args and **kwargs accept any additional arguments passed to evaluate().
        recorded["evaluate_labels"].extend(true_labels.tolist())

        # Execute the real evaluation function and record the resulting macro-F1.
        # The returned dictionary contains evaluation metrics, including "macro_f1".
        result = real_evaluate(true_labels, predictions, *args, **kwargs)
        recorded["val_f1s"].append(result["macro_f1"])

        return result

    # Temporarily replace the loss and evaluation functions with our spies.
    # monkeypatch automatically restores the original functions afterward.
    monkeypatch.setattr(F, "cross_entropy", spy_cross_entropy)
    monkeypatch.setattr(train_module, "evaluate", spy_evaluate)

    # Return the dictionary so tests can inspect what happened during training.
    return recorded

"""Check that class weights are passed to the training loss in every epoch."""
def test_loss_uses_class_weights(small_road_network, few_epochs, training_spies):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train the model while training_spies records the class weights used in the loss.
    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed = 0)

    # Verify that the loss function received class weights once per epoch.
    # Since few_epochs sets EPOCHS to 5, we expect five recorded entries.
    assert len(training_spies["loss_weights"]) == 5

    # None would indicate that no class weights were provided.
    # torch.equal() verifies that the weights match those calculated from the training labels.
    for weights in training_spies["loss_weights"]:
        assert weights is not None
        assert torch.equal(weights, class_weights)

"""Check that the selected best epoch has the highest validation macro-F1."""
def test_best_epoch_has_highest_validation_f1(small_road_network, few_epochs, training_spies):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train the MLP and record the macro-F1 score obtained after each epoch.
    # We only need the selected best epoch, so the predictions are ignored.
    _, best_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    val_f1s = training_spies["val_f1s"]

    # Verify that validation was performed after each of the five training epochs.
    assert len(val_f1s) == 5

    # max() finds the highest macro-F1, and index() finds its first occurrence.
    # Adding 1 converts the list index (starting at 0) into an epoch number (starting at 1).
    # If multiple epochs achieve the same score, the earliest one should be selected.
    assert best_epoch == val_f1s.index(max(val_f1s)) + 1

"""Give one segment a unique road-type label to detect label leakage."""
def give_unique_road_type(graph, segment, road_type):
    # This is a helper function, not a pytest test.
    # It assigns a road type to one segment that no other segment has.
    # If that road type appears in the wrong part of training, we know labels have leaked.

    # clone() creates a copy of the labels without modifying the original graph.
    changed_y = graph["y"].clone()
    changed_y[segment] = road_type

    # Create a new graph dictionary containing the modified labels.
    # Recalculate the usable road types so the new label is included in evaluation.
    changed_graph = {**graph, "y": changed_y}
    return changed_graph, labels_in_data(changed_y, graph["mask"])

"""Check that test labels are never used during training or validation."""
def test_test_labels_do_not_leak_into_training(small_road_network, few_epochs, training_spies):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Assign road type 2 to segment 10, which belongs to the test set.
    # No other segment has road type 2, so its appearance would reveal test-label leakage.
    graph, labels = give_unique_road_type(graph, segment = 10, road_type = 2)

    # Train the model while recording which labels are used for loss and validation.
    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed = 0)

    # Road type 2 must never appear in the training loss or validation evaluation.
    # Otherwise, information from the held-out test set would influence model development.
    assert 2 not in training_spies["loss_labels"]
    assert 2 not in training_spies["evaluate_labels"]

"""Check that validation labels are not used to calculate training loss."""
def test_validation_labels_do_not_leak_into_the_loss(
    small_road_network, few_epochs, training_spies,
):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Assign road type 3 to validation segment 3.
    # No other segment has road type 3, making it possible to detect its use.
    graph, labels = give_unique_road_type(graph, segment=3, road_type=3)

    # Train the model while recording the labels used in loss and evaluation.
    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # Validation labels must not be used to update the model's weights.
    # However, they must be included in validation to measure performance and select the best epoch.
    assert 3 not in training_spies["loss_labels"]
    assert 3 in training_spies["evaluate_labels"]

"""Check that MLP training uses the currently configured hyperparameters."""
def test_training_uses_current_mlp_settings(small_road_network, monkeypatch):
    # Get the artificial graph, training/validation masks, class weights, and usable labels.
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Temporarily set custom hyperparameters to verify that train_mlp() reads
    # its settings from mlp_config instead of using hardcoded values.
    # One epoch is sufficient because we only need to inspect model and optimizer creation.
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(mlp_config, "WORKING_SPACE", 11)
    monkeypatch.setattr(mlp_config, "DROPOUT", 0.2)
    monkeypatch.setattr(mlp_config, "LEARNING_RATE", 0.003)

    # Save the original MLP constructor and Adam optimizer before replacing them with spies.
    # We still want to create and train a real model, while recording the settings used.
    original_model = train_module.MLP
    original_adam = torch.optim.Adam

    model_settings = []
    optimizer_settings = []

    def record_model(road_features, working_space, road_types, dropout):
        # Record the dimensions and dropout passed when the MLP is created.
        # Then call the original constructor so normal training can continue.
        model_settings.append((road_features, working_space, road_types, dropout))
        return original_model(road_features, working_space, road_types, dropout)

    def record_optimizer(parameters, **kwargs):
        # Record the learning rate passed to Adam.
        # kwargs["lr"] retrieves the learning rate from the keyword arguments.
        optimizer_settings.append(kwargs["lr"])

        # Create the real optimizer using its original parameters.
        return original_adam(parameters, **kwargs)

    # Replace the constructors with spies that record settings while preserving their behavior.
    monkeypatch.setattr(train_module, "MLP", record_model)
    monkeypatch.setattr(torch.optim, "Adam", record_optimizer)

    # Execute the real training function using the temporary hyperparameters.
    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # Verify that the MLP was created with the correct input size, hidden size,
    # number of road types, and dropout probability.
    assert model_settings == [(graph["x"].shape[1], 11, len(ROAD_CLASSES), 0.2)]

    # Verify that Adam received the configured learning rate.
    assert optimizer_settings == [0.003]
