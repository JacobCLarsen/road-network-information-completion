# Tests for graphsage_model.py: the GraphSAGE model itself, without training. Training is tested in
# test_train_graphsage.py.

import pytest
import torch
from src.ml.models.graphsage.graphsage_model import GraphSAGE


# The model sizes used in every test: 4 features per segment, 64 hidden units and 21 road types
ROAD_FEATURES = 4
WORKING_SPACE = 64
ROAD_TYPES = 21


# A small hand-made road of four segments in a row: 0 - 1 - 2 - 3. Seen from segment 0, segment 1 is one step away,
# segment 2 two steps and segment 3 three steps.
# Each row of x is one segment's features: log(length), curvature, one-way, degree.
@pytest.fixture
def straight_road():
    x = torch.tensor([[4.2, 0.1, 1.0, 1.0],
                      [5.0, 0.0, 0.0, 2.0],
                      [3.8, 0.3, 0.0, 2.0],
                      [4.6, 0.2, 0.0, 1.0]])

    # Every connection is stored in both directions: 0 <-> 1, 1 <-> 2 and 2 <-> 3
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]])

    return x, edge_index


@pytest.fixture
def model():
    # Fixed seed, so the starting weights are the same every time the tests run
    torch.manual_seed(0)
    return GraphSAGE(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.5)


# Checks that the model gives one row per segment with one score per road type, the shape the training code expects
def test_output_shape(straight_road, model):
    x, edge_index = straight_road

    out = model(x, edge_index)

    # 4 segments, 21 road types
    assert out.shape == (4, ROAD_TYPES)


# Checks that the model outputs raw scores, not probabilities. F.cross_entropy in the training code applies
# softmax itself, so if the model also applied softmax, it would happen twice: nothing would crash, but the second
# softmax squashes the scores close together and the model would learn very slowly. If probabilities are needed, apply
# softmax outside the model: F.softmax(model(x, edge_index), dim=1).
def test_outputs_raw_scores_not_probabilities(straight_road, model):
    x, edge_index = straight_road
    model.eval()

    out = model(x, edge_index)

    # Probabilities are never negative, raw scores can be
    assert (out < 0).any()
    # Each row of probabilities sums to 1, raw scores do not. sum(dim=1) adds up each row, and allclose compares
    # with a small tolerance for rounding.
    assert not (torch.allclose(out.sum(dim=1), torch.ones(4)))


# Checks that every weight in both layers gets a gradient, so every weight can learn. If a layer were accidentally cut
# off from the output, its weights would never change during training, and nothing would crash.
def test_every_weight_gets_a_gradient(straight_road, model):
    x, edge_index = straight_road

    # Any number computed from the output works as a loss here. .sum() adds up every score in the output (4 segments
    # x 21 road types) into one number. backward() needs a single number to start from.
    loss = model(x, edge_index).sum()
    # .backward() goes back through the model from the loss and computes the gradient of every weight: which direction
    # each weight should move to lower the loss. PyTorch stores each weight's gradient in its .grad.
    loss.backward()

    # .named_parameters() gives every weight tensor of the model together with its name, e.g. "layer1.lin_l.weight".
    # Each SAGEConv layer has weights for the segment itself (lin_r) and for its neighbours (lin_l), plus a bias.
    for name, weights in model.named_parameters():
        # .grad is None if backward() never reached this weight, so it is not connected to the output at all.
        # The text after the comma is the message pytest shows if the assert fails. The f before the quotes makes it
        # an f-string: {name} is replaced by the weight's name, so you can see which weight failed.
        assert weights.grad is not None, f"{name} has no gradient"
        # .abs() makes every value positive and .sum() adds them up, so the total is only 0 if every value is 0.
        # Without .abs(), positive and negative gradients could cancel out to 0 even though they are not all 0.
        assert weights.grad.abs().sum() > 0, f"{name} has a gradient of only zeros"


# Checks that dropout is off in evaluation mode, so validation and test scores do not depend on chance.
# Calling the model twice is the point: with dropout on, every call drops different values, so two calls give different
# outputs. model.eval() only turns dropout off because the model passes training=self.training to F.dropout. If that
# were training=True or left out, eval() would do nothing, the two calls would differ and this test would fail.
def test_eval_mode_gives_same_output(straight_road, model):
    x, edge_index = straight_road

    model.eval()
    assert torch.equal(model(x, edge_index), model(x, edge_index))


# Checks that dropout is on in training mode: different values are dropped each time, so the output changes
def test_train_mode_uses_dropout(straight_road, model):
    x, edge_index = straight_road

    model.train()
    assert not (torch.equal(model(x, edge_index), model(x, edge_index)))


# Checks that the model uses its dropout setting. With dropout 0 nothing is dropped, so even in training mode the same
# input must give the same output. If the model ignored the setting and always dropped values, this would fail.
def test_dropout_zero_drops_nothing(straight_road):
    x, edge_index = straight_road
    torch.manual_seed(0)
    model = GraphSAGE(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.0)

    model.train()
    assert torch.equal(model(x, edge_index), model(x, edge_index))


# Checks that the model looks at a segment's neighbours, not only at the segment itself
def test_neighbour_changes_output(straight_road, model):
    x, edge_index = straight_road
    model.eval()
    before = model(x, edge_index)

    # Change only segment 1, which is a neighbour of segment 0
    changed_x = x.clone()
    changed_x[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    after = model(changed_x, edge_index)

    # Segment 0's own features did not change, so its output can only change because the model looks at its neighbours
    assert not (torch.equal(before[0], after[0]))


# Checks how far the model looks. Each of the two layers looks one step further along the roads, so a segment sees its
# neighbours and its neighbours' neighbours, but nothing further away. This also shows the model only uses the road
# connections: a segment that is not close by has no effect.
def test_looks_two_steps_away_but_not_three(straight_road, model):
    x, edge_index = straight_road
    model.eval()
    before = model(x, edge_index)

    # Change segment 2, two steps from segment 0: segment 0's output changes
    changed_x = x.clone()
    changed_x[2] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert not (torch.equal(before[0], model(changed_x, edge_index)[0]))

    # Change segment 3, three steps from segment 0: segment 0's output stays exactly the same
    changed_x = x.clone()
    changed_x[3] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.equal(before[0], model(changed_x, edge_index)[0])
