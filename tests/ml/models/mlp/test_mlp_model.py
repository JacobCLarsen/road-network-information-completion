# Tests for mlp_model.py: the MLP model itself, without training. Training is tested in test_train_mlp.py.

import pytest
import torch
from src.ml.models.mlp.mlp_model import MLP


# The model sizes used in every test: 4 features per segment, 64 hidden units and 21 road types
ROAD_FEATURES = 4
WORKING_SPACE = 64
ROAD_TYPES = 21


# Four hand-made road segments. Unlike GraphSAGE, the MLP gets no connections between segments: only one row of
# features per segment.
# Each row of x is one segment's features: log(length), curvature, one-way, degree.
@pytest.fixture
def segments():
    return torch.tensor([[4.2, 0.1, 1.0, 1.0],
                         [5.0, 0.0, 0.0, 2.0],
                         [3.8, 0.3, 0.0, 2.0],
                         [4.6, 0.2, 0.0, 1.0]])


@pytest.fixture
def model():
    # Fixed seed, so the starting weights are the same every time the tests run
    torch.manual_seed(0)
    return MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.5)


# Checks that the model gives one row per segment with one score per road type, the shape the training code expects
def test_output_shape(segments, model):
    out = model(segments)

    # 4 segments, 21 road types
    assert out.shape == (4, ROAD_TYPES)


# Checks that the model outputs raw scores, not probabilities. F.cross_entropy in the training code applies
# softmax itself, so if the model also applied softmax, it would happen twice: nothing would crash, but the second
# softmax squashes the scores close together and the model would learn very slowly. If probabilities are needed, apply
# softmax outside the model: F.softmax(model(x), dim=1).
def test_outputs_raw_scores_not_probabilities(segments, model):
    model.eval()

    out = model(segments)

    # Probabilities are never negative, raw scores can be
    assert (out < 0).any()
    # Each row of probabilities sums to 1, raw scores do not
    assert not (torch.allclose(out.sum(dim=1), torch.ones(4)))


# Checks that the same seed gives the same starting weights. The training code fixes the seed before creating the
# model, and relies on this to make every run with the same seed give the same result.
def test_same_seed_gives_same_starting_weights():
    torch.manual_seed(0)
    first = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.5)
    torch.manual_seed(0)
    second = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.5)

    for first_weights, second_weights in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_weights, second_weights)


# Checks that every weight in both layers gets a gradient, so every weight can learn. If a layer were accidentally cut
# off from the output, its weights would never change during training, and nothing would crash.
def test_every_weight_gets_a_gradient(segments, model):
    # Any single number computed from the output works as a loss here
    loss = model(segments).sum()
    loss.backward()

    # Each Linear layer has a weight matrix and a bias, e.g. "layer1.weight" and "layer1.bias"
    for name, weights in model.named_parameters():
        assert weights.grad is not None, f"{name} has no gradient"
        assert weights.grad.abs().sum() > 0, f"{name} has a gradient of only zeros"


# Checks that dropout is off in evaluation mode, so validation and test scores do not depend on chance.
# model.eval() only turns dropout off because the model passes training=self.training to F.dropout.
def test_eval_mode_gives_same_output(segments, model):
    model.eval()
    assert torch.equal(model(segments), model(segments))


# Checks that dropout is on in training mode: different values are dropped each time, so the output changes
def test_train_mode_uses_dropout(segments, model):
    model.train()
    assert not (torch.equal(model(segments), model(segments)))


# Checks that the model uses its dropout setting. With dropout 0 nothing is dropped, so even in training mode the same
# input must give the same output. If the model ignored the setting and always dropped values, this would fail.
def test_dropout_zero_drops_nothing(segments):
    torch.manual_seed(0)
    model = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout=0.0)

    model.train()
    assert torch.equal(model(segments), model(segments))


# Checks that the model looks at the segment itself: changing a segment's features changes its own scores
def test_own_features_change_output(segments, model):
    model.eval()
    before = model(segments)

    changed = segments.clone()
    changed[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    after = model(changed)

    assert not (torch.equal(before[0], after[0]))


# Checks the key difference from GraphSAGE: the MLP only looks at each segment on its own. Changing one segment must
# leave the scores of every other segment exactly the same. If the model mixed segments (for example by normalising
# over the whole batch), the comparison with GraphSAGE would no longer show how much the neighbours help.
def test_other_segments_do_not_change_output(segments, model):
    model.eval()
    before = model(segments)

    changed = segments.clone()
    changed[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    after = model(changed)

    # Segments 0, 2 and 3 did not change, so their scores stay exactly the same
    for segment in [0, 2, 3]:
        assert torch.equal(before[segment], after[segment]), f"segment {segment} changed"


# Checks that a segment gets the same scores whether it is scored alone or together with others, and that the order
# of the segments does not matter. The real graph has hundreds of thousands of segments, so this means the scores do
# not depend on which other segments happen to be in the data.
def test_scores_do_not_depend_on_other_segments_or_order(segments, model):
    model.eval()
    together = model(segments)

    # Scored alone: segments[2:3] keeps the row as a table of one row, so the shape is (1, 4) and not (4,)
    alone = model(segments[2:3])
    assert torch.allclose(alone[0], together[2])

    # Scored in reverse order: the scores come out in reverse order too. torch.flip(..., dims=[0]) reverses the rows.
    reversed_out = model(torch.flip(segments, dims=[0]))
    assert torch.allclose(torch.flip(reversed_out, dims=[0]), together)
