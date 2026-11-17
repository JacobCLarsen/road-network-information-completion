# Tests for train_mlp.py. Only train_mlp() is tested here, not prepare_data():
# prepare_data() loads the whole Beijing map from disk. That is slow and needs the data files, which makes it an
# integration test, not a unit test.


import pytest
import torch
import torch.nn.functional as F
import src.ml.models.mlp.train_mlp as train_module
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import labels_in_data
from src.ml.models.mlp.train_mlp import train_mlp


# The same small hand-made road network as the GraphSAGE training tests, so both models are tested on the same data.
# 12 segments: two separate roads of 6 segments each.
#
# 0 - 1 - 2 - 3 - 4 - 10    road type 0, feature 1.0
# 5 - 6 - 7 - 8 - 9 - 11    road type 1, feature -1.0 (segment 11: no road type, feature 0.0)
#
# 6 segments are for training (0-2 and 5-7) and 4 for validation (3-4 and 8-9). Like the real graph, it also has:
# - segment 10, a test segment: it has a road type, but is in neither the training nor the validation set
# - segment 11, an unlabelled segment (y = -1, mask = False)
# The MLP ignores the connections, but the graph still has them, because train_mlp gets the same graph as GraphSAGE.
# The feature tells the road type, so a working model should learn it easily.
@pytest.fixture
def small_road_network():
    # One row per segment: its feature (x) and its road type (y)
    x = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0],
                      [-1.0], [-1.0], [-1.0], [-1.0], [-1.0],
                      [1.0], [0.0]])
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1])

    # Every connection is stored in both directions
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 4, 10, 9, 11],
                               [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 9, 8, 10, 4, 11, 9]])

    # Segment 11 is the only unlabelled one
    mask = torch.tensor([True, True, True, True, True, True, True, True, True, True, True, False])
    graph = {"x": x, "edge_index": edge_index, "y": y, "mask": mask}

    # Segments 0-2 and 5-7 are for training, 3-4 and 8-9 for validation. Segments 10 and 11 are in neither.
    train_mask = torch.tensor([True, True, True, False, False, True, True, True, False, False, False, False])
    val_mask = torch.tensor([False, False, False, True, True, False, False, False, True, True, False, False])

    class_weights = compute_class_weights(y[train_mask], len(ROAD_CLASSES))
    labels = labels_in_data(y, mask)

    return graph, train_mask, val_mask, class_weights, labels


# Train for 5 epochs instead of the real number, so the tests are fast. monkeypatch resets it after the test.
# A test uses this fixture by listing few_epochs as a parameter, even though it never reads it.
@pytest.fixture
def few_epochs(monkeypatch):
    monkeypatch.setattr(config, "EPOCHS", 5)


# Checks that training gives one prediction for every segment, also when the graph has unlabelled segments, like the
# real Beijing graph. Unlabelled segments have y = -1. They are never in the training set, so they never reach the
# loss: if one did, cross_entropy would fail, because -1 is not a road type.
def test_one_prediction_per_segment(small_road_network, few_epochs):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    predictions, _ = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert predictions.shape == (12,)


# Checks that training is reproducible: the same seed must give the same predictions and best epoch. The dev run
# and the final test run train with the same seeds, so they rely on this to train exactly the same models.
def test_same_seed_gives_same_predictions(small_road_network, few_epochs):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    first_predictions, first_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)
    second_predictions, second_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert torch.equal(first_predictions, second_predictions)
    assert first_epoch == second_epoch


# Checks that training actually works. The model only learns from the training segments (0-2 and 5-7), so if it also
# gets the validation segments (3-4 and 8-9) right, it has learned the pattern "feature 1.0 is road type 0, feature
# -1.0 is road type 1" instead of memorising the training roads. If the loss, the optimiser or the masks were wrong,
# the model would not learn and this test would fail.
def test_predicts_all_validation_segments_correctly(small_road_network, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    # 5 epochs is too few to learn, so this test sets 50 itself instead of using few_epochs
    monkeypatch.setattr(config, "EPOCHS", 50)

    predictions, _ = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert torch.equal(predictions[val_mask], graph["y"][val_mask])


# Checks that the MLP really ignores the connections between segments: training on the same graph without any
# connections gives exactly the same result. This is what makes the MLP a fair "no neighbours" baseline for GraphSAGE.
# If train_mlp started using edge_index, the comparison would no longer show how much the neighbours help.
def test_connections_do_not_change_training(small_road_network, few_epochs):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    with_connections, with_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The same graph with no connections at all: an edge_index with 2 rows and 0 columns
    no_connections = {**graph, "edge_index": torch.empty((2, 0), dtype=torch.long)}
    without_connections, without_epoch = train_mlp(no_connections, train_mask, val_mask, class_weights, labels, seed=0)

    assert torch.equal(with_connections, without_connections)
    assert with_epoch == without_epoch


# Records what the loss and evaluate get during training, for the tests below. train_mlp only returns the
# predictions and the best epoch, which is not enough to see what happened inside: for example, one leaked label
# rarely changes which road type wins. Instead, the loss and evaluate are replaced by "spies" that write down what they
# get and then call the real function.
# Returns a dict of lists that fill up during training:
# - "loss_labels": every label the loss got
# - "loss_weights": the class weights the loss got, one entry per epoch
# - "evaluate_labels": every label evaluate got
# - "val_f1s": the validation macro-F1 evaluate computed, one entry per epoch
@pytest.fixture
def training_spies(monkeypatch):
    recorded = {"loss_labels": [], "loss_weights": [], "evaluate_labels": [], "val_f1s": []}

    real_cross_entropy = F.cross_entropy

    def spy_cross_entropy(scores, true_labels, **kwargs):
        recorded["loss_labels"].extend(true_labels.tolist())
        recorded["loss_weights"].append(kwargs.get("weight"))
        return real_cross_entropy(scores, true_labels, **kwargs)

    real_evaluate = train_module.evaluate

    def spy_evaluate(true_labels, predictions, *args, **kwargs):
        recorded["evaluate_labels"].extend(true_labels.tolist())
        result = real_evaluate(true_labels, predictions, *args, **kwargs)
        recorded["val_f1s"].append(result["macro_f1"])
        return result

    # Use the spies during the test only. monkeypatch puts the real functions back afterwards.
    monkeypatch.setattr(F, "cross_entropy", spy_cross_entropy)
    monkeypatch.setattr(train_module, "evaluate", spy_evaluate)

    return recorded


# Checks that the loss gets the class weights in every epoch. Without them nothing crashes, but mistakes on rare road
# types cost as little as mistakes on common ones, so on the real data the model would mostly predict residential.
def test_loss_uses_class_weights(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert len(training_spies["loss_weights"]) == 5
    for weights in training_spies["loss_weights"]:
        assert weights is not None
        assert torch.equal(weights, class_weights)


# Checks that best_epoch is the epoch with the highest validation macro-F1. If several epochs share the highest score,
# it must be the first of them, because a later epoch only replaces the best one when it is strictly better.
def test_best_epoch_has_highest_validation_f1(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    _, best_epoch = train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    val_f1s = training_spies["val_f1s"]
    assert len(val_f1s) == 5
    assert best_epoch == val_f1s.index(max(val_f1s)) + 1


# NOT A TEST!
# A helper function for the tests below.
# Gives one segment a road type that no other segment has, so that road type can only come from that segment.
# Returns a new graph and labels list, without changing the fixture's graph.
def give_unique_road_type(graph, segment, road_type):
    changed_y = graph["y"].clone()
    changed_y[segment] = road_type
    changed_graph = {**graph, "y": changed_y}
    return changed_graph, labels_in_data(changed_y, graph["mask"])


# Checks that test labels can never influence training. Only the training labels may be used for learning (the loss)
# and only the validation labels for choosing the epoch (evaluate). If a test label were used, the test scores would be
# too good, because the model would have seen the answers.
def test_test_labels_do_not_leak_into_training(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Test segment 10 gets road type 2, which no other segment has
    graph, labels = give_unique_road_type(graph, segment=10, road_type=2)

    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert 2 not in training_spies["loss_labels"]
    assert 2 not in training_spies["evaluate_labels"]


# Checks that validation labels are only used for choosing the epoch, never for learning. If the model learned from
# them, the validation scores would be too good, and the settings would be tuned on misleading numbers.
def test_validation_labels_do_not_leak_into_the_loss(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Validation segment 3 gets road type 3, which no other segment has
    graph, labels = give_unique_road_type(graph, segment=3, road_type=3)

    train_mlp(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert 3 not in training_spies["loss_labels"]
    assert 3 in training_spies["evaluate_labels"]
