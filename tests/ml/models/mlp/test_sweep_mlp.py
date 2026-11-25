# tests/ml/models/mlp/test_sweep_mlp.py

"""Test MLP hyperparameter sweeps without running full training experiments."""

"""Imports"""
import csv
import pytest
import torch
import src.ml.models.mlp.sweep_mlp as sweep_module
from src.ml import config
from src.ml.class_weights import compute_class_weights
from src.ml.data.road_graph_builder import ROAD_CLASSES
from src.ml.models.mlp import mlp_config

"""Create a small graph with fixed training, validation, and test partitions."""
@pytest.fixture
def sweep_graph():
    # pytest.fixture makes this graph available to any test that includes sweep_graph as an argument.
    # We use artificial data to test the sweep without loading or training on the Beijing dataset.

    # Create 12 segments with two arbitrary features each.
    # torch.ones((12, 2)) creates a tensor with 12 rows and 2 columns, all containing 1.0.
    # y contains road-type labels (0, 1, and 7), and mask indicates that all labels are usable.
    graph = {
        "x": torch.ones((12, 2)),
        "y": torch.tensor([0, 0, 0, 1, 1, 7, 0, 1, 7, 7, 7, 7]),
        "mask": torch.ones(12, dtype = torch.bool),
    }

    # Divide the 12 segments into three fixed partitions:
    # Training: segments 0-5, validation: segments 6-8, and testing: segments 9-11.
    # [True] * 6 creates six True values, while + joins the lists together.
    # True means that a segment belongs to the corresponding partition.
    train_mask = torch.tensor([True] * 6 + [False] * 6)
    val_mask = torch.tensor([False] * 6 + [True] * 3 + [False] * 3)
    test_mask = torch.tensor([False] * 9 + [True] * 3)

    # Return the graph, its partitions, and the three road types present in the data.
    return graph, train_mask, val_mask, test_mask, [0, 1, 7]

"""Check that validation scores are formatted with or without a standard deviation."""
def test_format_score_with_and_without_standard_deviation():
    # format_score() is our function in sweep_mlp.py that formats results to four decimal places.
    # When std is None, only the mean is displayed. Otherwise, the standard deviation is included.
    assert sweep_module.format_score({"mean": 0.5, "std": None}) == "0.5000"
    assert sweep_module.format_score({"mean": 0.5, "std": 0.0125}) == "0.5000 ± 0.0125"

"""Check that the sweep evaluates configurations on validation data and saves the best results first."""
def test_sweep_uses_validation_only_and_saves_sorted_results(monkeypatch, tmp_path, sweep_graph):
    # monkeypatch temporarily replaces functions or settings and restores them afterward.
    # tmp_path is a pytest fixture that creates a temporary directory for the CSV results.
    # sweep_graph is the artificial graph provided by our fixture above.
    graph, train_mask, val_mask, test_mask, labels = sweep_graph

    # Save the original hyperparameters so we can verify that the sweep restores them.
    # MLP-specific settings are stored in mlp_config, while class-weight power uses the shared config.
    original_settings = (
        mlp_config.WORKING_SPACE, mlp_config.DROPOUT,
        mlp_config.LEARNING_RATE, config.CLASS_WEIGHT_POWER,
    )

    # Run a small sweep with two configurations and two seeds.
    # Only CLASS_WEIGHT_POWER changes, while the other hyperparameters remain fixed.
    # This gives four training runs instead of the full set of experiments.
    monkeypatch.setattr(config, "MODEL_SEEDS", [0, 1])
    monkeypatch.setattr(sweep_module, "SWEEP", {
        "WORKING_SPACE": [12],
        "DROPOUT": [0.2],
        "LEARNING_RATE": [0.005],
        "CLASS_WEIGHT_POWER": [0.0, 0.5],
    })

    # Save results to a temporary CSV file instead of overwriting real experiment results.
    results_file = tmp_path / "mlp_results.csv"
    monkeypatch.setattr(sweep_module, "RESULTS_FILE", results_file)

    # Replace prepare_data() with a lambda, which is a small anonymous function.
    # It returns our artificial graph and fixed partitions without loading the Beijing dataset.
    # None replaces the initial class weights, which the sweep recalculates for each configuration.
    monkeypatch.setattr(sweep_module, "prepare_data", lambda: (
        graph, train_mask, val_mask, test_mask, None, labels
    ))

    # Record which class-weight powers and seeds are used during the sweep.
    # This allows us to verify that every intended configuration is evaluated.
    train_calls = []

    # Replace the real train_mlp() function with predictable behavior.
    # Instead of training a model, fake_train() checks its inputs and returns artificial predictions.
    def fake_train(prepared_graph, training, validation, weights, used_labels, seed):
        # Verify that training receives the correct graph, masks, and road-type labels.
        # torch.equal() compares the values of two PyTorch tensors.
        assert prepared_graph is graph
        assert torch.equal(training, train_mask)
        assert torch.equal(validation, val_mask)
        assert used_labels == labels

        # Verify that class weights are recalculated for the current weighting power.
        # Only training labels may be used, so validation and test labels cannot affect the weights.
        assert torch.equal(weights, compute_class_weights(
            graph["y"][train_mask], len(ROAD_CLASSES), config.CLASS_WEIGHT_POWER
        ))

        # Record the current weighting power and seed.
        # Expected combinations: (0.0, 0), (0.0, 1), (0.5, 0), and (0.5, 1).
        train_calls.append((config.CLASS_WEIGHT_POWER, seed))

        # torch.zeros_like() creates a tensor of zeros with the same shape as graph["y"].
        # This means every segment is initially predicted as road type 0.
        predictions = torch.zeros_like(graph["y"])

        # Make the configuration with power 0.5 predict every validation label correctly.
        # The configuration with power 0.0 keeps predicting road type 0, giving it a lower score.
        # This lets us check whether the sweep correctly ranks configurations by validation macro-F1.
        if config.CLASS_WEIGHT_POWER == 0.5:
            predictions[val_mask] = graph["y"][val_mask]

        # Return the artificial predictions and a predictable best epoch.
        # Seeds 0 and 1 produce best epochs 2 and 3, which we verify in the CSV.
        return predictions, seed + 2

    # Replace real training with fake_train() and execute the actual sweep.
    # The sweep still calculates metrics, compares configurations, and writes the CSV normally.
    monkeypatch.setattr(sweep_module, "train_mlp", fake_train)
    sweep_module.main()

    # Verify that both configurations were evaluated with both seeds in the expected order.
    assert train_calls == [(0.0, 0), (0.0, 1), (0.5, 0), (0.5, 1)]

    # Verify that all original hyperparameters were restored after the sweep finished.
    # Otherwise, later experiments could accidentally use settings from this test.
    assert (
        mlp_config.WORKING_SPACE, mlp_config.DROPOUT,
        mlp_config.LEARNING_RATE, config.CLASS_WEIGHT_POWER,
    ) == original_settings

    # Read the generated CSV file using csv.DictReader().
    # DictReader converts each row into a dictionary where the keys are the column names.
    with results_file.open(newline = "", encoding = "utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    # Verify that both configurations were saved and sorted by validation macro-F1.
    # The power 0.5 configuration should come first because its validation predictions were correct.
    assert len(rows) == 2
    assert rows[0]["class_weight_power"] == "0.5"
    assert float(rows[0]["macro_f1_mean"]) > float(rows[1]["macro_f1_mean"])

    # Verify that the CSV contains the correct seeds, best epochs, and per-seed scores.
    # Both seeds achieved perfect validation macro-F1 with power 0.5.
    assert rows[0]["model_seeds"] == "0 1"
    assert rows[0]["best_epochs"] == "2 3"
    assert rows[0]["macro_f1_by_seed"] == "1.0000 1.0000"

"""Check that temporary hyperparameters are restored even if training fails."""
def test_sweep_restores_configuration_when_training_fails(monkeypatch, tmp_path, sweep_graph):
    # Use the same artificial data as the previous test.
    graph, train_mask, val_mask, test_mask, labels = sweep_graph

    # Save the original settings before running a sweep that will deliberately fail.
    original_size = mlp_config.WORKING_SPACE
    original_power = config.CLASS_WEIGHT_POWER

    # Configure a small sweep that temporarily changes the hidden size and class-weight power.
    # The values are chosen to differ from the normal settings, making restoration detectable.
    monkeypatch.setattr(sweep_module, "SWEEP", {
        "WORKING_SPACE": [17],
        "CLASS_WEIGHT_POWER": [0.0],
    })

    # Use a temporary output file and replace data preparation with our artificial graph.
    results_file = tmp_path / "should_not_exist.csv"
    monkeypatch.setattr(sweep_module, "RESULTS_FILE", results_file)
    monkeypatch.setattr(sweep_module, "prepare_data", lambda: (
        graph, train_mask, val_mask, test_mask, None, labels
    ))

    # Simulate a failure during training by raising RuntimeError.
    # *args accepts any positional arguments passed by the sweep without using them.
    def fail_training(*args):
        raise RuntimeError("Training failed")

    # Temporarily replace train_mlp() with fail_training() to simulate a training failure.
    # This lets us verify that the sweep restores its settings even when an error occurs.
    monkeypatch.setattr(sweep_module, "train_mlp", fail_training)

    # pytest.raises() checks that the expected error occurs.
    # sweep_module.main() runs the real sweep, but training has been replaced with
    # fail_training(), which raises the error so we can test the sweep's failure handling.
    with pytest.raises(RuntimeError, match = "Training failed"):
        sweep_module.main()

    # Verify that the failure did not leave temporary hyperparameters in memory.
    # No results file should exist because the sweep failed before saving its CSV.
    assert mlp_config.WORKING_SPACE == original_size
    assert config.CLASS_WEIGHT_POWER == original_power
    assert not results_file.exists()

"""Check that invalid sweep settings are rejected before loading any data."""
# pytest.mark.parametrize() runs the same test with multiple sets of arguments.
# Here, we test an unknown hyperparameter name and a hyperparameter with no values.
# Each case provides the invalid settings and the expected error message.
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"UNKNOWN_SETTING": [1]}, "Unknown setting: UNKNOWN_SETTING"),
        ({"DROPOUT": []}, "No values provided for DROPOUT"),
    ],
)
def test_sweep_rejects_invalid_settings_before_loading_data(monkeypatch, settings, message):
    # Replace SWEEP with the current invalid configuration.
    # pytest automatically supplies settings and message from the parameterized cases.
    monkeypatch.setattr(sweep_module, "SWEEP", settings)

    # Create a function that deliberately raises an error if data preparation is attempted.
    # This ensures invalid settings are rejected before loading the Beijing dataset.
    def fail_if_data_loaded():
        raise AssertionError("Invalid sweeps must not load the dataset")

    # Temporarily replace prepare_data() with fail_if_data_loaded().
    # If the sweep attempts to load data before validating its settings, the test will fail.
    monkeypatch.setattr(sweep_module, "prepare_data", fail_if_data_loaded)

    # pytest.raises() checks that the expected ValueError occurs.
    # sweep_module.main() runs the real sweep, which should detect the invalid settings
    # and raise the error before reaching the mocked prepare_data() function.
    with pytest.raises(ValueError, match = message):
        sweep_module.main()
