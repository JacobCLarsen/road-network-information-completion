# tests/ml/models/mlp/test_mlp_model.py

"""Test the MLP model architecture and behavior without running training."""

"""Imports"""
import pytest
import torch
from src.ml.models.mlp.mlp_model import MLP

# Define the model dimensions used throughout the tests.
# Each segment has four features, the hidden layer has 64 neurons,
# and the model predicts one of eight possible road types.
ROAD_FEATURES = 4
WORKING_SPACE = 64
ROAD_TYPES = 8

"""Create four artificial road segments with predefined features."""
@pytest.fixture
def segments():
    # pytest.fixture makes these segments available to tests that include segments as an argument.
    # We use artificial data to test the MLP without loading the Beijing dataset.

    # torch.tensor() converts the lists into a PyTorch tensor.
    # Each row represents one road segment with four features:
    # log(length), curvature, one-way, and graph degree.
    # Unlike GraphSAGE, the MLP receives no information about connections between segments.
    return torch.tensor([
        [4.2, 0.1, 1.0, 1.0],
        [5.0, 0.0, 0.0, 2.0],
        [3.8, 0.3, 0.0, 2.0],
        [4.6, 0.2, 0.0, 1.0],
    ])

"""Create an MLP with fixed dimensions and reproducible starting weights."""
@pytest.fixture
def model():
    # torch.manual_seed() fixes PyTorch's random seed so the model starts
    # with the same randomly initialized weights whenever this fixture is created.
    torch.manual_seed(0)

    # Create an MLP with four input features, 64 hidden neurons, eight output classes,
    # and a dropout probability of 0.5.
    return MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout = 0.5)

"""Check that the model returns one score per road type for every segment."""
def test_output_shape(segments, model):
    # Pass the four artificial segments through the MLP to calculate their output scores.
    out = model(segments)

    # The output must contain four rows (segments) and eight columns (road types).
    # This is the shape expected by the training and evaluation code.
    assert out.shape == (4, ROAD_TYPES)

"""Check that the MLP outputs raw scores instead of probabilities."""
def test_outputs_raw_scores_not_probabilities(segments, model):
    # model.eval() disables dropout so the output is not affected by randomness.
    model.eval()

    # Pass the four segments through the MLP to calculate their road-type scores.
    # Calling model(segments) automatically executes the model's forward() method.
    out = model(segments)

    # Raw scores (logits) can be negative, while probabilities cannot.
    # .any() checks whether at least one output score is negative.
    assert (out < 0).any()

    # Probabilities must sum to 1 for each segment, whereas raw scores do not.
    # sum(dim = 1) adds the road-type scores for each segment, and torch.ones(4)
    # creates four ones for comparison. allclose() allows small rounding differences.
    assert not torch.allclose(out.sum(dim = 1), torch.ones(4)) 

    # NOTE: 
    # dim = 1: Sums horizontally (across the columns) 
    # dim = 0: Sums vertically (down the rows)

    # The training code uses cross-entropy loss, which already handles the softmax calculation.
    # Applying softmax inside the MLP would therefore be unnecessary and could affect training.

"""Check that creating models with the same random seed produces identical starting weights."""
def test_same_seed_gives_same_starting_weights():
    # Create two independent MLPs using the same random seed.
    # Both models should receive exactly the same initial parameter values.
    torch.manual_seed(0)
    first = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout = 0.5)

    torch.manual_seed(0)
    second = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout = 0.5)

    # parameters() returns the model's trainable weights and biases.
    # zip() pairs each parameter tensor from the first model with the corresponding one from the second.
    # torch.equal() verifies that both tensors contain exactly the same values.
    for first_weights, second_weights in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_weights, second_weights)

"""Check that the model's trainable parameters receive gradients during backpropagation."""
def test_every_weight_gets_a_gradient(segments, model):
    # Calculate the model's output and combine all scores into one artificial loss.
    # We do not need a real loss function here because we only want to test gradient flow.
    loss = model(segments).sum()

    # backward() calculates gradients for the model's trainable parameters.
    # These gradients are necessary for updating weights during training.
    loss.backward()

    # named_parameters() returns each parameter tensor together with its name.
    # For example, a linear layer typically has both a weight matrix and a bias vector.
    for name, weights in model.named_parameters():
        # Verify that PyTorch calculated a gradient for the parameter tensor.
        # If grad is None, the parameter was not connected to the loss calculation.
        assert weights.grad is not None, f"{name} has no gradient"

        # abs().sum() adds the absolute values of all gradients in the tensor.
        # A positive result confirms that at least one parameter value has a nonzero gradient.
        assert weights.grad.abs().sum() > 0, f"{name} has a gradient of only zeros"

"""Check that the MLP produces identical outputs when evaluated repeatedly."""
def test_eval_mode_gives_same_output(segments, model):
    # model.eval() disables dropout, making the model's output deterministic.
    # This is important because validation and test results should not depend on random dropout.
    model.eval()

    # torch.equal() checks that repeated predictions contain exactly the same values.
    assert torch.equal(model(segments), model(segments))

"""Check that dropout changes the model's output during training."""
def test_train_mode_uses_dropout(segments, model):
    # model.train() enables training behavior, including dropout.
    # Dropout randomly sets some hidden activations to zero during each forward pass.
    model.train()

    # With dropout enabled, repeated forward passes should normally produce different outputs.
    # This verifies that the model actually applies dropout during training.
    assert not torch.equal(model(segments), model(segments))

"""Check that a dropout probability of zero disables dropout completely."""
def test_dropout_zero_drops_nothing(segments):
    # Create a separate MLP with dropout set to 0.0.
    # This means no hidden activations should be randomly removed.
    torch.manual_seed(0)
    model = MLP(ROAD_FEATURES, WORKING_SPACE, ROAD_TYPES, dropout = 0.0)

    # Even in training mode, repeated predictions must now be identical.
    # This verifies that the model respects its configured dropout probability.
    model.train()
    assert torch.equal(model(segments), model(segments))

"""Check that changing a segment's own features changes its predicted scores."""
def test_own_features_change_output(segments, model):
    # Disable dropout so that changes in predictions come from the input features.
    model.eval()
    before = model(segments)

    # clone() creates a separate copy of the original feature tensor.
    # Change only the features of segment 0 without modifying the fixture's original data.
    changed = segments.clone()
    changed[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Calculate new predictions after modifying the segment's features.
    after = model(changed)

    # The scores for segment 0 should change because its own input features changed.
    assert not torch.equal(before[0], after[0])

"""Check that changing one segment does not affect predictions for other segments."""
def test_other_segments_do_not_change_output(segments, model):
    # The MLP processes each segment independently, unlike GraphSAGE,
    # which uses information from neighboring road segments.
    model.eval()
    before = model(segments)

    # Change the features of segment 1 while keeping all other segments unchanged.
    changed = segments.clone()
    changed[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    after = model(changed)

    # Segments 0, 2, and 3 were not modified, so their output scores must remain identical.
    # If their predictions change, the model may be mixing information between segments.
    for segment in [0, 2, 3]:
        assert torch.equal(before[segment], after[segment]), f"segment {segment} changed"

"""Check that predictions do not depend on other segments or their ordering."""
def test_scores_do_not_depend_on_other_segments_or_order(segments, model):
    # Disable dropout so that predictions are not influenced by randomness.
    # First, calculate predictions for all four segments together.
    model.eval()
    together = model(segments)

    # Test whether segment 2 receives the same scores when evaluated alone.
    # segments[2:3] selects only segment 2 but preserves the two-dimensional tensor shape (1, 4).
    # torch.allclose() allows small numerical differences caused by floating-point calculations.
    alone = model(segments[2:3])
    assert torch.allclose(alone[0], together[2])

    # torch.flip() reverses the order of the segments along dimension 0 (the rows).
    # After reversing the output rows again, the predictions should match the original results.
    reversed_out = model(torch.flip(segments, dims = [0]))
    assert torch.allclose(torch.flip(reversed_out, dims = [0]), together)
