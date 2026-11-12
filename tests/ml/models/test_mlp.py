# tests/ml/models/test_mlp.py

import torch
import torch.nn.functional as F

from src.ml.models.mlp.mlp_model import MLP

# Artificial road segments: [log(length), curvature, one_way, degree].
# The first three look like residential roads (class 6), the last three like motorways (class 0).
FEATURES = torch.tensor([[3.0, 0.15, 0.0, 2.0],
                         [3.2, 0.12, 0.0, 3.0],
                         [3.5, 0.10, 0.0, 2.0],
                         [5.8, 0.01, 1.0, 6.0],
                         [6.0, 0.02, 1.0, 7.0],
                         [6.2, 0.01, 1.0, 5.0]])
LABELS = torch.tensor([6, 6, 6, 0, 0, 0])


def test_output_shape():
    road_types = 21
    model = MLP(road_features=4, working_space=64, road_types=road_types, dropout=0.5)
    out = model(FEATURES)

    # One row per road segment, one score per road type
    assert out.shape == (FEATURES.shape[0], road_types)


def test_same_seed_gives_same_starting_weights():
    # The run scripts fix the seed before creating the model, so every run with the same seed gives the same result
    torch.manual_seed(0)
    first = MLP(4, 16, 21, 0.5)
    torch.manual_seed(0)
    second = MLP(4, 16, 21, 0.5)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_dropout_only_in_training_mode():
    model = MLP(4, 64, 21, dropout=0.5)

    # Evaluation mode turns dropout off, so the same input always gives the same scores
    model.eval()
    assert torch.equal(model(FEATURES), model(FEATURES))

    # Training mode turns dropout on, so the scores change from call to call
    model.train()
    torch.manual_seed(0)
    first = model(FEATURES)
    torch.manual_seed(1)
    second = model(FEATURES)
    assert not torch.equal(first, second)


def test_learns_separable_road_types():
    # A few full-batch training steps on clearly different road types should give the correct predictions
    torch.manual_seed(0)
    model = MLP(4, 16, 21, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(FEATURES), LABELS)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(FEATURES).argmax(dim=1)
    assert predictions.tolist() == LABELS.tolist()
