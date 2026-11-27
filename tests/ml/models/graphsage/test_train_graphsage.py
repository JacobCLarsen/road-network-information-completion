# Tests for train_graphsage.py. Only train_graphsage() is tested here, not prepare_data():
# prepare_data() loads the whole Beijing map from disk. That is slow and needs the data files, which makes it an
# integration test, not a unit test.


import pytest
import torch
import torch.nn.functional as F
import src.ml.models.graphsage.train_graphsage as train_module
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.evaluation import labels_in_data
from src.ml.models.graphsage import graphsage_config as sage_config
from src.ml.models.graphsage.train_graphsage import train_graphsage


# A small hand-made road network of 12 segments: two separate roads of 6 segments each.
#
# 0 - 1 - 2 - 3 - 4 - 10    road type 0, feature 1.0
# 5 - 6 - 7 - 8 - 9 - 11    road type 1, feature -1.0 (segment 11: no road type, feature 0.0)
#
# 6 segments are for training (0-2 and 5-7) and 4 for validation (3-4 and 8-9). Like the real graph, it also has:
# - segment 10, a test segment: it has a road type, but is in neither the training nor the validation set
# - segment 11, an unlabelled segment (y = -1, mask = False), which is only there as a neighbour
# The feature tells the road type, so a working model should learn it easily.
@pytest.fixture
def small_road_network():
    # One row per segment: its feature (x) and its road type (y)
    x = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0],
                      [-1.0], [-1.0], [-1.0], [-1.0], [-1.0],
                      [1.0], [0.0]])
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1])

    # The first row is "from" and the second row is "to", so each column is one connection. Every connection is
    # stored in both directions: 0 -> 1 and 1 -> 0, 1 -> 2 and 2 -> 1, and so on, ending with 4 <-> 10 and 9 <-> 11.
    # There is no connection between 4 and 5. GraphSAGE mixes each segment's features with its neighbours', so a
    # connection would give segments 4 and 5 a mix of both road types, and the learning test could fail by chance.
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


# Train for 5 epochs instead of the real number, so the tests are fast, and with 3 groups (2 training segments each,
# because the network has 6 training segments), so we know how many steps there are. monkeypatch resets both after
# the test.
# A test uses this fixture by listing few_epochs as a parameter. The editor greys the parameter out because the test
# never reads it, but pytest still runs the fixture before the test, and that is what changes EPOCHS.
@pytest.fixture
def few_epochs(monkeypatch):
    monkeypatch.setattr(config, "EPOCHS", 5)
    monkeypatch.setattr(sage_config, "GROUPS", 3)


# Checks that training gives one prediction for every segment, also when the graph has unlabelled segments, like the
# real Beijing graph. Unlabelled segments have y = -1. They are never in the training set, so they never reach the
# loss: if one did, cross_entropy would fail, because -1 is not a road type.
def test_one_prediction_per_segment(small_road_network, few_epochs):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # train_graphsage returns predictions and best_epoch. The _ is best_epoch, which this test does not need.
    predictions, _ = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The network has 12 segments, and the unlabelled segment 11 also gets a prediction, so there must be 12
    # predictions. .shape is a tuple with one number per dimension, and predictions has one dimension of 12 values.
    # The comma makes (12,) a tuple: (12) would just be the number 12.
    assert predictions.shape == (12,)


# Checks that training is reproducible: the same seed must give the same predictions and best epoch. The dev run
# and the final test run train with the same seeds, so they rely on this to train exactly the same models.
def test_same_seed_gives_same_predictions(small_road_network, few_epochs):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # Train twice with the same seed
    first_predictions, first_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)
    second_predictions, second_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The seed fixes the starting weights and dropout, so both runs must give exactly the same result
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

    # The _ is best_epoch, which this test does not need
    predictions, _ = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # After training, every validation segment gets the right road type.
    # predictions[val_mask] keeps only the values where val_mask is True, so segments 3, 4, 8 and 9: the predictions
    # should be [0, 0, 1, 1]. graph["y"][val_mask] picks the true road types of the same segments, also [0, 0, 1, 1].
    assert torch.equal(predictions[val_mask], graph["y"][val_mask])


# Records what the loss and evaluate get during training, for the tests below. train_graphsage only returns the
# predictions and the best epoch, which is not enough to see what happened inside: for example, one leaked label
# rarely changes which road type wins. Instead, the loss and evaluate are replaced by "spies" that write down what they
# get and then call the real function.
# Returns a dict of lists that fill up during training:
# - "loss_labels": every label the loss got
# - "loss_weights": the class weights the loss got, one entry per group step
# - "loss_scores": the scores the loss got, one entry per group step
# - "evaluate_labels": every label evaluate got
# - "val_f1s": the validation macro-F1 evaluate computed, one entry per epoch
@pytest.fixture
def training_spies(monkeypatch):
    recorded = {"loss_labels": [], "loss_weights": [], "loss_scores": [], "evaluate_labels": [], "val_f1s": []}

    # The spy for the loss: write down the labels, then compute the real loss.
    # kwargs means keyword arguments: arguments passed by name, like weight=class_weights. In the definition, **kwargs
    # collects them into a dict, e.g. {"weight": class_weights}. In the call, **kwargs unpacks that dict back into
    # named arguments, so the real loss still gets its class weights without the spy having to know which ones exist.
    real_cross_entropy = F.cross_entropy

    def spy_cross_entropy(scores, true_labels, **kwargs):
        recorded["loss_labels"].extend(true_labels.tolist())
        # .get gives None if no weight was passed, instead of an error
        recorded["loss_weights"].append(kwargs.get("weight"))
        # .detach() takes the scores out of the training, so storing them does not keep the whole computation alive
        recorded["loss_scores"].append(scores.detach())
        return real_cross_entropy(scores, true_labels, **kwargs)

    # The spy for evaluate: write down the labels, then compute the real scores.
    # *args is the same idea for arguments passed without a name: one * collects them into a tuple. evaluate is called
    # with labels as an extra unnamed argument, and *args passes it on.
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


# Checks that GraphSAGE really uses the connections between segments during training: training on the same graph
# without any connections gives different scores. This is the opposite of the MLP, which must ignore them. If
# train_graphsage stopped passing edge_index to the model, GraphSAGE would just be an MLP and the comparison between the
# two would show nothing.
# The scores are compared instead of the predictions, because on this small network the feature alone tells the road
# type, so both runs would likely predict the same.
def test_connections_change_training(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The same graph with no connections at all: an edge_index with 2 rows and 0 columns
    no_connections = {**graph, "edge_index": torch.empty((2, 0), dtype=torch.long)}
    train_graphsage(no_connections, train_mask, val_mask, class_weights, labels, seed=0)

    # loss_scores has one entry per group step, so 15 per run (few_epochs sets 5 epochs and 3 groups). [0] is the
    # first step of the run with connections and [15] the first step of the run without. Same seed, same starting
    # weights and same groups: only the connections differ.
    scores = training_spies["loss_scores"]
    assert len(scores) == 30
    assert not (torch.equal(scores[0], scores[15]))


# Checks that the loss gets the class weights in every step (one step per group, so 3 per epoch). Without them nothing crashes, but mistakes on rare road
# types cost as little as mistakes on common ones, so on the real data the model would mostly predict residential.
def test_loss_uses_class_weights(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # One entry per group in every epoch (5 epochs * 3 groups), and every one is exactly the class weights passed in
    assert len(training_spies["loss_weights"]) == 15
    for weights in training_spies["loss_weights"]:
        assert weights is not None
        assert torch.equal(weights, class_weights)


# Checks that best_epoch is the epoch with the highest validation macro-F1. If several epochs share the highest score,
# it must be the first of them, because a later epoch only replaces the best one when it is strictly better.
# Not tested: that the returned predictions come from the best epoch and not the last one. On this small network the
# validation macro-F1 reaches 1.0 after a few epochs and stays there, so the best and last epochs predict the same and
# such a test would pass even if the predictions came from the last epoch.
def test_best_epoch_has_highest_validation_f1(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    # The _ is the predictions, which this test does not need
    _, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # evaluate is called once per epoch, so val_f1s has one score per epoch (few_epochs sets EPOCHS to 5)
    val_f1s = training_spies["val_f1s"]
    assert len(val_f1s) == 5

    # .index gives the position of the first highest score, counted from 0. Epochs are counted from 1, hence + 1.
    assert best_epoch == val_f1s.index(max(val_f1s)) + 1


# NOT A TEST!
# A helper function for the tests below.
# Gives one segment a road type that no other segment has, so that road type can only come from that segment.
# Returns a new graph and labels list. clone() makes a copy of y, so the fixture's y is not changed.
def give_unique_road_type(graph, segment, road_type):
    changed_y = graph["y"].clone()
    changed_y[segment] = road_type
    # ** copies every key and value of graph into a new dict, then "y" is set to changed_y. When a key appears twice,
    # the last one wins. It makes a new dict, so the fixture's graph is not changed (graph["y"] = changed_y would).
    changed_graph = {**graph, "y": changed_y}
    return changed_graph, labels_in_data(changed_y, graph["mask"])


# Checks that test labels can never influence training. Only the training labels may be used for learning (the loss)
# and only the validation labels for choosing the epoch (evaluate). If a test label were used, the test scores would be
# too good, because the model would have seen the answers.
def test_test_labels_do_not_leak_into_training(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Test segment 10 gets road type 2, which no other segment has
    graph, labels = give_unique_road_type(graph, segment=10, road_type=2)

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The test segment's road type was used neither for learning nor for choosing the epoch
    assert 2 not in training_spies["loss_labels"]
    assert 2 not in training_spies["evaluate_labels"]


# Checks that validation labels are only used for choosing the epoch, never for learning. If the model learned from
# them, the validation scores would be too good, and the settings would be tuned on misleading numbers.
def test_validation_labels_do_not_leak_into_the_loss(small_road_network, few_epochs, training_spies):
    graph, train_mask, val_mask, class_weights, _ = small_road_network

    # Validation segment 3 gets road type 3, which no other segment has
    graph, labels = give_unique_road_type(graph, segment=3, road_type=3)

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The validation segment's road type was not used for learning, but was used for choosing the epoch
    assert 3 not in training_spies["loss_labels"]
    assert 3 in training_spies["evaluate_labels"]


# Group hiding

# A spy for neighbour_label_shares: writes down the known mask of every call, then computes the real shares.
@pytest.fixture
def known_masks(monkeypatch):
    recorded = []
    real_shares = train_module.neighbour_label_shares

    def spy_shares(edge_index, y, known_mask, num_classes):
        recorded.append(known_mask.clone())
        return real_shares(edge_index, y, known_mask, num_classes)

    monkeypatch.setattr(train_module, "neighbour_label_shares", spy_shares)
    return recorded


# Checks that every epoch hides every training segment exactly once. The first call is for validation and the
# predictions and uses all training labels. After that there are 3 calls per epoch, one per group.
def test_every_training_segment_is_hidden_once_per_epoch(small_road_network, few_epochs, known_masks):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert torch.equal(known_masks[0], train_mask)
    assert len(known_masks) == 1 + 5 * 3
    for epoch in range(5):
        hidden_count = torch.zeros(len(train_mask), dtype=torch.long)
        for known in known_masks[1 + epoch * 3: 1 + (epoch + 1) * 3]:
            # the hidden segments in this step: training segments that are not known
            hidden = train_mask & ~known
            assert hidden.sum().item() == 2
            hidden_count = hidden_count + hidden.long()
        # every training segment hidden exactly once, and nothing else ever hidden
        assert torch.equal(hidden_count, train_mask.long())


# Checks that the neighbour label columns only ever use training labels, never validation or test labels.
def test_neighbour_labels_only_use_training_labels(small_road_network, few_epochs, known_masks):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    for known in known_masks:
        # known must be a subset of the training segments
        assert not (known & ~train_mask).any()


# Checks that the loss is only computed on the hidden group, so a segment is never trained on while its own
# label is in the input.
def test_loss_only_on_the_hidden_group(small_road_network, few_epochs, known_masks, training_spies):
    graph, train_mask, val_mask, class_weights, labels = small_road_network

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # 2 labels per step (the hidden group), 15 steps
    assert len(training_spies["loss_labels"]) == 2 * 15


def test_groups_must_be_at_least_two(small_road_network, few_epochs, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    monkeypatch.setattr(sage_config, "GROUPS", 1)

    with pytest.raises(ValueError, match="GROUPS"):
        train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)


# Checks that train_graphsage reads its settings from graphsage_config when it is called, not hard-coded values. The
# sweep relies on this: it changes graphsage_config between runs and expects the next model to use the new values.
def test_training_uses_current_graphsage_settings(small_road_network, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(sage_config, "WORKING_SPACE", 11)
    monkeypatch.setattr(sage_config, "DROPOUT", 0.3)
    monkeypatch.setattr(sage_config, "LEARNING_RATE", 0.003)
    monkeypatch.setattr(sage_config, "WEIGHT_DECAY", 1e-4)

    # Spies on the model and the optimiser: write down the settings, then create the real ones
    real_model = train_module.GraphSAGE
    real_adam = torch.optim.Adam
    model_settings = []
    optimizer_settings = []

    def spy_model(road_features, working_space, road_types, dropout):
        model_settings.append((road_features, working_space, road_types, dropout))
        return real_model(road_features, working_space, road_types, dropout)

    def spy_adam(parameters, **kwargs):
        optimizer_settings.append((kwargs["lr"], kwargs["weight_decay"]))
        return real_adam(parameters, **kwargs)

    monkeypatch.setattr(train_module, "GraphSAGE", spy_model)
    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # The model gets the normal features plus one neighbour label column per road type
    assert model_settings == [(graph["x"].shape[1] + len(ROAD_CLASSES), 11, len(ROAD_CLASSES), 0.3)]
    assert optimizer_settings == [(0.003, 1e-4)]


# Checks that with VALIDATE_EVERY = 2 the validation set is only checked after epochs 2 and 4, and after the last
# epoch (5) even though 5 is not a multiple of 2. Without the last check, the last epochs could never be chosen.
def test_validates_every_n_epochs_and_after_the_last(small_road_network, few_epochs, training_spies, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    monkeypatch.setattr(sage_config, "VALIDATE_EVERY", 2)

    _, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    # Training still runs every epoch (3 group steps each), only the check is skipped
    assert len(training_spies["loss_weights"]) == 5 * 3
    assert len(training_spies["val_f1s"]) == 3
    # The best epoch can only be one of the checked epochs
    checked_epochs = [2, 4, 5]
    assert best_epoch == checked_epochs[training_spies["val_f1s"].index(max(training_spies["val_f1s"]))]


# Checks that the returned predictions come from the best epoch, not the last one. On the small network the real
# validation macro-F1 only goes up, so here evaluate is replaced by a stand-in that scores the epochs the other way
# round: the first epoch gets the highest score and every later epoch a lower one, so the best epoch is epoch 1.
# The predictions would be the same at every epoch on this small network, so the scores are compared instead: a spy on
# the model writes down its scores every time it runs in evaluation mode. The scores after training must be exactly
# the scores of epoch 1, which only happens if the weights of the best epoch were loaded back.
def test_predictions_come_from_the_best_epoch(small_road_network, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    monkeypatch.setattr(config, "EPOCHS", 5)

    checked = []

    def reversed_evaluate(true_labels, predictions, *args, **kwargs):
        checked.append(True)
        # 1.0 for the first epoch, then lower and lower
        return {"macro_f1": 1.0 - len(checked) / 100}

    monkeypatch.setattr(train_module, "evaluate", reversed_evaluate)

    # The model's scores every time it runs in evaluation mode: once per epoch, then once more after training
    eval_scores = []
    real_model = train_module.GraphSAGE

    def spy_model(*args):
        model = real_model(*args)
        # A forward hook runs after every call of the model and gets its output
        model.register_forward_hook(
            lambda module, inputs, output: None if module.training else eval_scores.append(output.detach().clone()))
        return model

    monkeypatch.setattr(train_module, "GraphSAGE", spy_model)

    predictions, best_epoch = train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0)

    assert best_epoch == 1
    # 5 epochs plus the run after training
    assert len(eval_scores) == 6
    first_epoch, last_epoch, after_training = eval_scores[0], eval_scores[-2], eval_scores[-1]
    assert torch.equal(after_training, first_epoch)
    assert not torch.equal(after_training, last_epoch)
    assert torch.equal(predictions, after_training.argmax(dim=1))


# Checks that the training curve gets one row per epoch, with the loss of every epoch and the validation macro-F1 of
# the validated epochs only. The run scripts save it in the database, so a missing or wrong row would give a wrong
# curve in the API.
def test_history_has_one_row_per_epoch(small_road_network, few_epochs, training_spies, monkeypatch):
    graph, train_mask, val_mask, class_weights, labels = small_road_network
    monkeypatch.setattr(sage_config, "VALIDATE_EVERY", 2)

    history = []
    train_graphsage(graph, train_mask, val_mask, class_weights, labels, seed=0, history=history)

    assert [row["epoch"] for row in history] == [1, 2, 3, 4, 5]
    # Validated after epochs 2 and 4, and after the last epoch (5)
    assert [row["val_macro_f1"] for row in history] == [None, training_spies["val_f1s"][0], None,
                                                        training_spies["val_f1s"][1], training_spies["val_f1s"][2]]
    # The loss is a plain number (the average over the groups), so it can be saved in the database
    assert all(isinstance(row["train_loss"], float) for row in history)
